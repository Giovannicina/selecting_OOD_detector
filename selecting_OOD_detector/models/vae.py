"""
Module providing an implementation of a Variational Auto-Encoder.
"""

# STD
from typing import List, Tuple, Optional

# EXT
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from selecting_OOD_detector.models.novelty_estimator import NoveltyEstimator

# Default training hyperparameters
DEFAULT_LEARNING_RATE: float = 1e-2
DEFAULT_RECONSTR_ERROR_WEIGHT: float = 1e20
DEFAULT_N_VAE_SAMPLES: int = 100


class Encoder(nn.Module):
    """The encoder module, which encodes an input into the latent space.

    Parameters
    ----------
    hidden_sizes: List[int]
        A list with the sizes of the hidden layers.
    input_size: int
        The input dimensionality.
    latent_dim: int
        The size of the latent space.
    """

    def __init__(self, hidden_sizes: List[int], input_size: int, latent_dim: int):
        super().__init__()
        architecture = [input_size] + hidden_sizes
        self.layers = []

        for l, (in_dim, out_dim) in enumerate(zip(architecture[:-1], architecture[1:])):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.LeakyReLU())

        self.hidden = nn.Sequential(*self.layers)
        self.mean = nn.Linear(architecture[-1], latent_dim)
        self.log_var = nn.Linear(architecture[-1], latent_dim)

    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform forward pass of encoder. Returns mean and standard deviation corresponding to
        an independent Normal distribution.

        Parameters
        ----------
        input_tensor: torch.Tensor
            The input to the encoder.
        """
        h = self.hidden(input_tensor)
        mean = self.mean(h)
        log_var = self.log_var(h)
        std = torch.sqrt(torch.exp(log_var))

        return mean, std


class Decoder(nn.Module):
    """
    The decoder module, which decodes a sample from the latent space back to the space of
    the input data.

    Parameters
    ----------
    hidden_sizes: List[int]
        A list with the sizes of the hidden layers.
    input_size: int
        The dimensionality of the input
    latent_dim: int
        The size of the latent space.
    """

    def __init__(self, hidden_sizes: List[int], input_size: int, latent_dim: int):
        super().__init__()
        architecture = [latent_dim] + hidden_sizes
        self.layers = []

        for l, (in_dim, out_dim) in enumerate(zip(architecture[:-1], architecture[1:])):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.LeakyReLU())

        self.hidden = nn.Sequential(*self.layers)
        self.mean = nn.Linear(architecture[-1], input_size)
        self.log_var = nn.Linear(architecture[-1], input_size)

    def forward(
            self, latent_tensor: torch.Tensor, reconstruction_mode: str = "mean"
    ) -> torch.Tensor:
        """Perform forward pass of decoder. Returns mean and standard deviation corresponding
        to an independent Normal distribution.

        Parameters
        ----------
        latent_tensor: torch.Tensor
            A sample from the latent space, which has to be decoded.
        reconstruction_mode: str
            Specify the way that a sample should be reconstructed. 'mean' simply returns the mean of p(x|z), 'sample'
            samples from the same distribution.
        """
        assert reconstruction_mode in ["mean", "sample"], (
            "Invalid reconstruction mode given, must be 'mean' or "
            f"'sample', '{reconstruction_mode}' found."
        )

        h = self.hidden(latent_tensor)
        mean = self.mean(h)
        log_var = self.log_var(h)
        std = torch.sqrt(torch.exp(log_var))

        # Just return the mean
        if reconstruction_mode == "mean":
            return mean

        # Sample
        else:
            eps = torch.randn(mean.shape)
            return mean + eps * std

    def reconstruction_error(
            self, input_tensor: torch.Tensor, latent_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the log probability of the original data sample under p(x|z).

        Parameters
        ----------
        input_tensor: torch.Tensor
            Original data sample.
        latent_tensor: torch.Tensor
            A sample from the latent space, which has to be decoded.

        Returns
        -------
        reconstr_error: torch.Tensor
            Log probability of the input under the decoder's distribution.
        """
        h = self.hidden(latent_tensor)
        mean = self.mean(h)
        log_var = self.log_var(h)
        std = torch.sqrt(torch.exp(log_var))

        distribution = dist.independent.Independent(dist.normal.Normal(mean, std), 0)

        # calculating losses
        reconstr_error = -distribution.log_prob(input_tensor).sum(dim=1)

        return reconstr_error


class VAEModule(nn.Module):
    """The Pytorch module of a Variational Autoencoder, consisting of an equally-sized encoder and
    decoder. This module works for continuous distributions. In case of discrete distributions,
    it has to be adjusted (outputting a Bernoulli distribution instead of independent Normal).

    Parameters
    ----------
    input_size: int
        The dimensionality of the input, assumed to be a 1-d vector.
    hidden_sizes: List[int]
        A list of integers, representing the hidden dimensions of the encoder and decoder. These
        hidden dimensions are the same for the encoder and the decoder.
    latent_dim: int
        The dimensionality of the latent space.
    """

    def __init__(self, hidden_sizes: List[int], input_size: int, latent_dim: int):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = Encoder(hidden_sizes, input_size, latent_dim)
        self.decoder = Decoder(hidden_sizes, input_size, latent_dim)

    def forward(
            self,
            input_tensor: torch.Tensor,
            reconstr_error_weight: float,
            beta: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform an encoding and decoding step and return the
        reconstruction error, KL-divergence and negative average elbo for the given batch.

        Parameters
        ----------
        input_tensor: torch.Tensor
            The input to the VAE.
        reconstr_error_weight: float
            A factor which is multiplied with the reconstruction error, to weigh this term in
            the overall loss function.
        beta: float
            Weighting term for the KL divergence.

        Returns
        -------
        reconstr_error: torch.Tensor
            The reconstruction error.
        kl: torch.Tensor
            The KL-divergence.
        average_negative_elbo: torch.Tensor
            The negative ELBO averaged over the batch.
        """

        input_tensor = input_tensor.float()
        # encoding
        mean, std = self.encoder(input_tensor)
        eps = torch.randn(mean.shape)
        z = mean + eps * std

        # decoding
        reconstr_error = self.decoder.reconstruction_error(input_tensor, z)
        d = mean.shape[1]

        # Calculating the KL divergence of the two independent Gaussians (closed-form solution)
        kl = 0.5 * torch.sum(
            std - torch.ones(d) - torch.log(std + 1e-8) + mean * mean, dim=1
        )
        average_negative_elbo = torch.mean(
            reconstr_error_weight * reconstr_error + kl * beta
        )

        return reconstr_error, kl, average_negative_elbo

    def get_reconstruction_error_grad(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Return the gradient of log p(x|z).

        Parameters
        ----------
        input_tensor: torch.Tensor
            Input for which the gradient of the reconstruction error should be computed.

        Returns
        -------
        torch.Tensor
            Gradient of reconstruction error w.r.t. the input.
        """
        model_state = self.encoder.training
        self.encoder.train()
        self.decoder.train()

        input_tensor = input_tensor.float()
        input_tensor.requires_grad = True

        # Encoding
        h = self.encoder.hidden(input_tensor)
        mean = self.encoder.mean(h)

        # Decoding
        reconstr_error = self.decoder.reconstruction_error(input_tensor, mean)
        # Compute separate grad for each bach instance
        reconstr_error.backward(gradient=torch.ones(reconstr_error.shape))
        grad = input_tensor.grad

        # Reset model state to what is was before
        self.encoder.training = model_state
        self.decoder.training = model_state

        return grad

    def get_reconstruction_grad_magnitude(
            self, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Retrieve the l2-norm of the gradient of log(x|z) w.r.t to the input.

        Parameters
        ----------
        input_tensor: torch.Tensor
            Input for which the magnitude of the gradient w.r.t. the reconstruction error should be computed.

        Returns
        -------
        torch.Tensor
            Magnitude of gradient of reconstruction error wr.t. the input.
        """
        norm = torch.norm(self.get_reconstruction_error_grad(input_tensor), dim=1)

        return norm


class VAE(NoveltyEstimator):
    """
    The VAE class that handles training and reconstruction.

    Parameters
    ----------
    input_size: int
        The dimensionality of the input
    hidden_sizes: List[int]
        A list with the sizes of the hidden layers.
    latent_dim: int
        The size of the latent space.
    beta: float
        Weighting term for the KL-divergence.
    anneal: bool
        Option to indicate whether KL-divergence should be annealed.
    """

    def __init__(
            self,
            hidden_sizes: List[int],
            input_size: int,
            latent_dim: int,
            beta: float = 1.0,
            anneal: bool = True,
            lr: float = DEFAULT_LEARNING_RATE,
            reconstr_error_weight: float = DEFAULT_RECONSTR_ERROR_WEIGHT,
    ):
        super().__init__(model_type="density_estimator")

        self.model = VAEModule(
            input_size=input_size, hidden_sizes=hidden_sizes, latent_dim=latent_dim
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.reconstr_error_weight = reconstr_error_weight
        self.beta = beta
        self.anneal = anneal
        self.current_epoch = 0

    def train(self,
              X_train: np.ndarray,
              **kwargs):
        """
        Train an AE for a number of epochs. Overrides the abstract method of Novelty Estimator.

        Parameters
        ----------
        X_train: np.ndarray
             Training data.

        **kwargs:
            batch_size: int
                The batch size, default 32
            n_epochs: int
                The number of epochs to train, default 6.
        """
        batch_size = kwargs.get("batch_size", 32)
        n_epochs = kwargs.get("n_epochs", 6)

        self.current_epoch = 0
        self._initialize_dataloaders(train_data=X_train, val_data=None, batch_size=batch_size)

        for epoch in range(n_epochs):
            self.model.train()
            self._epoch_iter(self.train_data, epoch, n_epochs)

            if self.val_data is not None:
                self.model.eval()
                self._epoch_iter(self.val_data, epoch, n_epochs)

    def get_novelty_score(self,
                          X: np.ndarray,
                          **kwargs):
        """
        Apply the novelty estimator to obtain a novelty score for the data.

        Parameters
         ----------
        X: np.array

        **kwargs:
            scoring_function: str
                Indicates which scoring fuction to use.
                mplemented options: "reconstr_err", "latent_prob", "latent_prior_prob", "reconstr_err_grad"

        Returns
         -------
        np.array
            Novelty scores for each sample.
        """

        scoring_function = kwargs.get("scoring_function", "reconstr_err")
        assert scoring_function in ["reconstr_err", "latent_prob", "latent_prior_prob", "reconstr_err_grad"]

        if scoring_function == "reconstr_err":
            return self._get_reconstr_error(X)

        elif scoring_function == "latent_prob":
            return self._get_latent_prob(X)

        elif scoring_function == "latent_prior_prob":
            return self._get_latent_prior_prob(X)

        elif scoring_function == "reconstr_err_grad":
            return self._get_reconstruction_grad_magnitude(X)

    def _initialize_dataloaders(
            self, train_data: np.ndarray, val_data: Optional[np.ndarray], batch_size: int
    ):
        """
        Initialize the dataloaders from original numpy data.

        Parameters
        ----------
        train_data: np.ndarray
            The data to train on.
        val_data: np.ndarray
            The data to validate on.
        batch_size: int
            The batch size to be used for training.
        """
        train_dataset = torch.from_numpy(train_data).float()
        self.train_data = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size
        )
        if val_data is not None:
            val_dataset = torch.from_numpy(val_data).float()
            self.val_data = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size
            )
        else:
            self.val_data = None

    def _epoch_iter(
            self, data: torch.utils.data.DataLoader, current_epoch: int, n_epochs: int
    ) -> float:
        """Iterate through the data once and return the average negative ELBO. If the train data
        is fed,the model parameters are updated. If the validation data is fed, only the average
        elbo is calculated and no parameter update is performed.

        Parameters
        ----------
        data: torch.utils.data.DataLoader
            The dataloader of the train or validation set.
        current_epoch: int
            Number of current epoch.
        n_epochs: int
            Total number of epochs.

        Returns
        -------
        average_epoch_elbo: float
            The negative ELBO averaged over the epoch.
        """

        average_epoch_elbo, i = 0, 0

        for i, batch in enumerate(tqdm(data)):

            if self.anneal and self.model.training:
                beta = self._get_beta(
                    target_beta=self.beta,
                    current_epoch=current_epoch,
                    current_iter=i,
                    n_epochs=n_epochs,
                    n_iters=len(data),
                )

            else:
                beta = self.beta

            _, _, average_negative_elbo = self.model(
                batch, reconstr_error_weight=self.reconstr_error_weight, beta=beta
            )
            average_epoch_elbo += average_negative_elbo

            if self.model.training:

                if torch.isnan(average_negative_elbo):
                    raise ValueError("ELBO is nan.")

                average_negative_elbo.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 4)
                self.optimizer.step()
                self.optimizer.zero_grad()

        average_epoch_elbo = average_epoch_elbo / (i + 1)

        return average_epoch_elbo

    @staticmethod
    def _get_beta(
            target_beta: float,
            current_epoch: int,
            current_iter: int,
            n_epochs: int,
            n_iters: int,
            saturation_percentage: float = 0.4,
    ) -> float:
        """
        Get the current beta term.

        Parameters
        ----------
        target_beta: float
            Target value for beta.
        current_epoch: int
            Current epoch number.
        current_iter: int
            Number of interations in current epoch.
        n_epochs: int
            Total number of epochs.
        n_iters:
            Number of iterations per epoch.
        saturation_percentage: float
            Percentage of total iterations after which the target_beta value should be reached.

        Returns
        -------
        float
            Annealed beta value.
        """
        total_iters = n_epochs * n_iters
        current_total_iter = current_epoch * n_iters + current_iter
        annealed_beta = (
                min(current_total_iter / (saturation_percentage * total_iters), 1)
                * target_beta
        )

        return annealed_beta

    def _get_reconstr_error(
            self, data: np.ndarray, n_samples: int = DEFAULT_N_VAE_SAMPLES
    ) -> np.ndarray:
        """Calculate the reconstruction error for some data (assumed to be a numpy array).
        The reconstruction error is averaged over a number of samples.

        Parameters
        ----------
        data: np.ndarray
            The data of which we want to know the reconstruction error.
        n_samples: int, default 10
            The number of samples to take to calculate the average reconstruction error.

        Returns
        -------
        avg_reconstruction_error: np.ndarray
            The average reconstruction error for each item in the data.
        """
        self.model.eval()
        reconstructions = []

        for i in range(n_samples):
            reconstr_error, _, _ = self.model(
                torch.from_numpy(data), reconstr_error_weight=self.reconstr_error_weight
            )
            reconstructions.append(reconstr_error.unsqueeze(0).detach().numpy())

        concatenated_rec = np.concatenate(reconstructions, axis=0)
        avg_reconstruction_error = np.mean(concatenated_rec, axis=0)

        return avg_reconstruction_error

    def _get_latent_encoding(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode the data to the latent space. The latent representation is defined by a
        mean and standard deviation corresponding to an independent Normal distribution.

        Parameters
        ----------
        data: np.ndarray
            The data for which we want to get the latent encodings.

        Returns
        -------
        z: np.ndarray
            The latent encoding of the data.
        """
        self.model.eval()
        mean, std = self.model.encoder(torch.from_numpy(data).unsqueeze(0).float())
        mean = mean.squeeze(0).detach().numpy()
        std = std.squeeze(0).detach().numpy()

        eps = np.random.randn(*mean.shape)
        z = mean + eps * std

        return z

    def _get_latent_prior_prob(self, data: np.ndarray) -> np.ndarray:
        """
        Get the probability of the latent representation corresponding to an input according
        to the latent space prior p(z).

        Parameters
        ----------
        data: np.ndarray
            The data for which we want to get the latent probabilities.

        Returns
        -------
        np.ndarray
            Log probabilities of latent representations.
        """
        self.model.eval()
        mean, _ = self.model.encoder(torch.from_numpy(data).float())

        # For VAE, the latent space is an isotropic gaussian
        distribution = dist.independent.Independent(dist.normal.Normal(0, 1), 0)
        latent_prob = distribution.log_prob(mean).sum(dim=1).detach().numpy()

        return latent_prob

    def _get_latent_prob(self, data: np.ndarray) -> np.ndarray:
        """
        Get the probability of the latent representation corresponding to an input according
        to q(z|x).

        Parameters
        ----------
        data: np.ndarray
            The data for which we want to get the latent probabilities.

        Returns
        -------
        np.ndarray
            Log probabilities of latent representations.
        """
        self.model.eval()
        mean, std = self.model.encoder(torch.from_numpy(data).float())

        # For VAE, the latent space is an isotropic gaussian
        distribution = dist.independent.Independent(dist.normal.Normal(mean, std), 0)
        latent_prob = distribution.log_prob(mean).sum(dim=1).detach().numpy()

        return latent_prob

    def _get_reconstruction_grad_magnitude(self, data: np.ndarray) -> np.ndarray:
        """
        Retrieve the l2-norm of the gradient of log(x|z) w.r.t to the input.

        Parameters
        ----------
        data: data: np.ndarray
            Input for which the magnitude of the gradient w.r.t. the reconstruction error should be computed.

        Returns
        -------
        data: np.ndarray
            Magnitude of gradient of reconstruction error wr.t. the input.
        """
        data = torch.from_numpy(data)
        grad_magnitude = self.model.get_reconstruction_grad_magnitude(data)
        grad_magnitude = grad_magnitude.detach().numpy()

        return grad_magnitude
