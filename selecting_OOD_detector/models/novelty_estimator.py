"""
A module implementing an abstract class of a Novelty Estimator. Base class for all novelty estimation models.
"""

from abc import ABC, abstractmethod

import numpy as np


class NoveltyEstimator(ABC):
    """
    Abstract class to be implemented by each model. Has two functions to be implemented: train and get_novelty_score
    """

    def __init__(self,
                 model_type="density_estimator"):
        """
        Parameters
        ----------
        model_type: str
            Model type indicates whether the novelty estimator predicts labels ("discriminator") or
            learns density of features ("density_estimator")
        """
        assert model_type in ["discriminator", "density_estimator"]
        self.model_type = model_type

    @abstractmethod
    def train(self,
              X_train: np.ndarray,
              **kwargs):
        """
        Train the novelty estimator.

        Parameters
        ----------
        X_train:  np.ndarray
            Training data.

        **kwargs:
            y_train: np.ndarray
                 Labels corresponding to the training data.
            X_val: np.ndarray
                Validation data.
            y_val: np.ndarray
                 Labels corresponding to the validation data.
            batch_size: int
                The batch size.
            n_epochs: int
                The number of training epochs.
        """
        pass

    @abstractmethod
    def get_novelty_score(self,
                          X: np.ndarray,
                          **kwargs
                          ) -> np.ndarray:
        """
        Apply the novelty estimator to obtain a novelty score for the data.

        Parameters
        ----------
        X: np.ndarray
            Samples to be scored.

        **kwargs:
            scoring_function: str
                If a novelty estimator has more than one way of scoring samples, indicates which function to use.

        Returns
        -------
        np.ndarray
            Novelty scores for each sample.
        """
        pass
