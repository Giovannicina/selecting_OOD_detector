from abc import ABC
from typing import Optional

import pandas as pd

from selecting_OOD_detector.utils.model_training import get_novelty_estimators


class BasePipeline(ABC):
    """
    Base pipeline to fit models on provided data.
    """

    def __init__(self,
                 model_selection: Optional[set] = None,
                 ):
        """

        Parameters
        ----------
        model_selection: set
            Define which models to train, e.g. {"PPCA", "LOF", "VAE"}. If selection is not provided, all available
            models are used.
        """

        self.model_selection = model_selection
        self.novelty_estimators = {}

    def _fit(self,
             X_train: pd.DataFrame,
             y_train: pd.DataFrame = None,
             n_trials: int = 5,
             ):
        """
        Fits models on training data with n_trials different runs. Returns a nested dictionary with keys
        being the trial number.
        """

        for i in range(n_trials):
            print(f"\n\n{i + 1}/{n_trials} trials:")

            print("\n\tTraining novelty estimators...")
            self.novelty_estimators[i] = \
                get_novelty_estimators(X_train=X_train, y_train=y_train, model_selection=self.model_selection)
