import os
from abc import ABC
from typing import Optional
import json

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
             hyperparameters_dir: Optional[str] = None
             ):
        """
        Fits models on training data with n_trials different runs. Returns a nested dictionary with keys
        being the trial number.
        """

        hyperparameters_init, hyperparameters_train = self._load_hyperparameters(path=hyperparameters_dir)

        for i in range(n_trials):
            print(f"\n\n{i + 1}/{n_trials} trials:")

            print("\n\tTraining novelty estimators...")
            self.novelty_estimators[i] = \
                get_novelty_estimators(X_train=X_train,
                                       y_train=y_train,
                                       model_selection=self.model_selection,
                                       hyperparameters_init=hyperparameters_init,
                                       hyperparameters_train=hyperparameters_train)

    @staticmethod
    def _load_hyperparameters(path: Optional[str] = None):
        """
        Loads json files of hyperparameters. If no path is provided, loads default hyperparameters saved in
        data/hyperparameters/default.

        """
        if path is None:
            path = "../data/hyperparameters/default/"

        with open(os.path.join(path, "init"), 'rb') as file:
            hyperparameters_init = json.load(file)

        with open(os.path.join(path, "train"), 'rb') as file:
            hyperparameters_train = json.load(file)

        return hyperparameters_init, hyperparameters_train
