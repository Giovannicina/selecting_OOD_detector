"""
Module providing an implementation of an a Masked Autoregressive Normalizing Flow using nflows package:
https://github.com/bayesiains/nflows/

"""

import numpy as np
import torch
from nflows.flows.autoregressive import MaskedAutoregressiveFlow
from torch import optim

from selecting_OOD_detector.models.novelty_estimator import NoveltyEstimator

EARLY_STOPPING_LIMIT = 2


class Flow(NoveltyEstimator):
    """
    Implements a Masked Autoregressive Flow (MAF).
    """

    def __init__(self,
                 input_size: int,
                 hidden_features: int = 512,
                 num_layers: int = 5,
                 num_blocks_per_layer=2,
                 batch_norm_between_layers: bool = True,
                 lr: float = 1e-3,
                 **kwargs
                 ):
        """
        input_size: int
            Dimensionality of the input.
        hidden_features: int
            Number of features in the hidden layers of transformations.
        num_layers: int
            Number of layers of transformations to be used.
        num_blocks_per_layer: int
            Number of blocks to be used in each transformation layer.
        batch_norm_between_layers: bool
            Specifies whether to use batch normalization between hidden layers.

        **kwargs:
        use_residual_blocks: bool
            Specifies whether to use residual blocks containing masked linear modules. Note that residual blocks can't
             be used with random masks. Default value is True.
        use_random_masks=False,
            Specifies whether to use a random mask inside a linear module with masked weigth matrix. Note that residual
             blocks can't be used with random masks. Default value is False.
        use_random_permutations: bool
            Specifies whether features are shuffled at random after each transformation.
            Default value is False.
        activation
            Activation function in hidden layers. Default value torch.nn.functional.relu
        dropout_probability: float
            Dropout rate in hidden layers. Default value is 0.0
        batch_norm_within_layers: bool
            Specifies whether to use batch normalization within hidden layers. Default value is False.
        """
        super().__init__(model_type="density_estimator")

        self.model = MaskedAutoregressiveFlow(features=input_size,
                                              hidden_features=hidden_features,
                                              num_layers=num_layers,
                                              num_blocks_per_layer=num_blocks_per_layer,
                                              batch_norm_between_layers=batch_norm_between_layers,
                                              **kwargs)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self,
              X_train: np.ndarray,
              **kwargs):
        """
        Train the novelty estimator.

        Parameters
        ----------
        X_train: np.array
            Training data.
        **kwargs:
            batch_size: int
                The batch size, default 128
            n_epochs: int
                The number of training epochs, default 30
        """

        batch_size = kwargs.get("batch_size", 128)
        n_epochs = kwargs.get("n_epochs", 30)

        ds_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float())
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)

        for epoch in range(n_epochs):
            for batch in dl_train:
                self.optimizer.zero_grad()
                x = batch[0]
                loss = -self.model.log_prob(inputs=x).mean()
                loss.backward()
                self.optimizer.step()

    def get_novelty_score(self,
                          X: np.ndarray,
                          **kwargs):
        """
        Apply the novelty estimator to obtain a novelty score for the data.
        Returns scores that indicate negative log probability for each sample under the learned distribution.

        Parameters
        ----------
        X: np.ndarray
            Samples to be scored.

        Returns
        -------
        np.ndarray
        Novelty scores for each sample.
        """

        # TODO: log_prob does not work for only one sample
        single_sample = False
        if X.ndim == 1 or X.shape[0] == 1:
            single_sample = True
            X = np.stack([X, X]).reshape(2, -1)

        X = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            log_prob = self.model.log_prob(X).numpy()

        if any(np.isnan(log_prob)) or any(np.isinf(log_prob)):
            print("\t Warning: encountered NaN or Inf in the log probabilites (Flow)")

        if single_sample:
            return - log_prob[0]

        return - log_prob
