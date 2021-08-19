"""
Very brief wrapper class for the scikit-learn LOF class to make it a bit more consistent with other models.
"""
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from selecting_OOD_detector.models.novelty_estimator import NoveltyEstimator


class LOF(LocalOutlierFactor, NoveltyEstimator):
    """
    LOF measures the local density of a given sample with respect to its closest neighbors.
    """

    def __init__(self, **kwargs):
        LocalOutlierFactor.__init__(self, n_neighbors=kwargs.get("n_neighbors"), algorithm="brute",  novelty=True)
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
        Apply the novelty estimator to obtain a novelty score for the data.
        Returns negative scores obtained from score_samples function. score_samples function assigns the large
        values to inliers and small values to outliers. Here, the negative  is used to get large values for outliers.

        Parameters
        ----------
        X: np.ndarray
            Samples to be scored.

        Returns
        -------
        np.ndarray
            Novelty scores for each sample.
        """
        return - self.score_samples(X)
