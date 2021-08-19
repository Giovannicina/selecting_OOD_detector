"""
Very brief wrapper class for the scikit-learn PCA class to make it a bit more consistent with other models.
"""
import numpy as np
from sklearn.decomposition import PCA

from selecting_OOD_detector.models.novelty_estimator import NoveltyEstimator


class PPCA(PCA, NoveltyEstimator):

    def __init__(self, **kwargs):
        PCA.__init__(self, n_components=kwargs.get("n_components"))
        NoveltyEstimator.__init__(self, model_type="density_estimator")

    def train(self,
              X_train: np.ndarray,
              **kwargs):
        """
        Train the novelty estimator.

        Parameters
        ----------
        X_train: np.array
            Training data.

        """
        super().fit(X_train)

    def get_novelty_score(self,
                          X: np.ndarray,
                          **kwargs,
                          ) -> np.ndarray:
        """
        Apply the novelty estimator to obtain a novelty score for the data..

        Parameters
        ----------
        X: np.ndarray
            Samples to be scored.

        Returns
        -------
        np.ndarray
            Novelty scores for each sample.
        """
        return -self.score_samples(X)
