"""TVAE module. Adapted from https://github.com/sdv-dev/CTGAN/blob/master/ctgan/synthesizers/tvae.py"""

import numpy as np
import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential, Dropout
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from ctgan.data_transformer import DataTransformer


class Encoder(Module):
    """Encoder for the TVAE.

    Args:
        data_dim (int):
            Dimensions of the data.
        compress_dims (tuple or list of ints):
            Size of each hidden layer.
        embedding_dim (int):
            Size of the output vector.
        (NEW!) encoder_dropout (int):
            Dropout rate for the encoder. Defaults to 0 (no dropout).            
    """

    def __init__(self, data_dim, compress_dims, embedding_dim, encoder_dropout): # NEW: encoder_dropout argument
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [
                Linear(dim, item),
                ReLU(), 
                Dropout(encoder_dropout) # NEW: encoder_dropout
            ]
            dim = item

        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input_):
        """Encode the passed `input_`."""
        feature = self.seq(input_)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(Module):
    """Decoder for the TVAE.

    Args:
        embedding_dim (int):
            Size of the input vector.
        decompress_dims (tuple or list of ints):
            Size of each hidden layer.
        data_dim (int):
            Dimensions of the data.
        (NEW!) decoder_dropout (int):
            Dropout rate for the decoder. Defaults to 0 (no dropout).
    """

    def __init__(self, embedding_dim, decompress_dims, data_dim, decoder_dropout): # NEW: decoder_dropout argument
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU(), Dropout(decoder_dropout)] # NEW: decoder_dropout
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input_):
        """Decode the passed `input_`."""
        return self.seq(input_), self.sigma


def _loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor):
    st = 0
    loss = []
    
    # reconstruction loss (loss_1)
    for column_info in output_info:
        for span_info in column_info:
            if span_info.activation_fn != 'softmax':
                ed = st + span_info.dim
                std = sigmas[st]
                eq = x[:, st] - torch.tanh(recon_x[:, st])
                loss.append((eq ** 2 / 2 / (std ** 2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed

            else:
                ed = st + span_info.dim
                loss.append(cross_entropy(
                    recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
                st = ed

    assert st == recon_x.size()[1]
    
    # KL divergence with normal gaussian -> regularization loss (loss_2)
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())

    # return loss_1, loss_2
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]


class TVAE():
    """TVAE."""

    def __init__(
        self,
        embedding_dim=128, # size of hidden dimension
        compress_dims=(128, 128), # encoder dimensions
        decompress_dims=(128, 128), # decoder dimensions
        encoder_dropout=0, # NEW: encoder_dropout
        decoder_dropout=0, # NEW: decoder_dropout
        seed=2023, # NEW: manual training seed
        weight_decay=1e-5, # NEW: renamed "l2scale" argument to "weight_decay" -> L2 regularization
        batch_size=500,
        epochs=300,
        loss_factor=2, # trade-off between loss 1 (reconstruction loss) and loss 2 (regularization loss) 
        verbose=False, # NEW: whether to have print statements for progress results
        cuda=True
    ):

        self.embedding_dim = embedding_dim 
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.encoder_dropout = encoder_dropout # NEW: encoder_dropout
        self.decoder_dropout = decoder_dropout # NEW: decoder_dropout

        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs
        self.verbose = verbose

        # NEW: set manual seed for reproducability
        torch.manual_seed(seed)
        np.random.seed(seed)

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

    @staticmethod
    def name() -> str: # NEW: add name to module
        return 'custom_tvae'        
        
    def fit(self, train_data, discrete_columns=(), eval_epochs=[], scoring_method=None, val_data=None):
        """Fit the TVAE Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
            (NEW!) eval_epochs (list-like): 
                List of epochs at which to sample synthetic data and evaluate its quality 
                compared to validation data using the scoring_method. Used for efficient
                tuning of the number of epochs as a hyperparameter. 
            (NEW!) scoring_method (function): 
                Calculates a similarity metric between ground-truth validation data (val_data) and 
                generated synthetic data. 
            (NEW!) val_data (pandas.DataFrame): 
                Validation dataset for calculating synthetic data quality throughout training. 
        """
        
        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)
        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        data_dim = self.transformer.output_dimensions
        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim, self.encoder_dropout).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim, self.decoder_dropout).to(self._device)
        optimizerAE = Adam(
            list(encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.weight_decay)

        val_scores = [] # NEW: keep track of the validation scores for the epochs in eval_epochs

        for i in range(self.epochs):
            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()
                real = data[0].to(self._device)
                mu, std, logvar = encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = self.decoder(emb)
                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar,
                    self.transformer.output_info_list, self.loss_factor
                )
                loss = loss_1 + loss_2
                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

            if self.verbose:
                print(f'Epoch {i+1}, Reconstruction loss: {loss_1.detach().cpu(): .4f}, '
                      f'KL Div Regularization loss: {loss_2.detach().cpu(): .4f}',
                      flush=True)

            # NEW: keep track of the validation scores for the epochs listed in eval_epochs
            if i+1 in eval_epochs: 
                
                # put modules into eval mode
                encoder.eval()
                self.decoder.eval()

                # turn off grad calculation
                with torch.no_grad():
                    synthetic_data = self.sample(len(val_data))
                    score = scoring_method(val_data, synthetic_data, max_bins=20)
                    val_scores.append(score)

                # put modules back into train mode
                encoder.train()
                self.decoder.train()

        return val_scores

    def sample(self, n, seed=2023): # TODO: note that seed argument is not yet used in sample() - currently, the same seed for training is used
        """Sample data similar to the training data.

        Args:
            n (int):
                Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        self.decoder.eval()

        steps = n // self.batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)
            fake, sigmas = self.decoder(noise)
            fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]
        return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU')."""
        self._device = device
        self.decoder.to(self._device)