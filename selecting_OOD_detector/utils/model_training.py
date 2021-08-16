import os
from typing import Optional
import pickle

import pandas as pd

from selecting_OOD_detector.models.novelty_estimators_info import (IMPLEMENTED_MODELS,
                                                                   HYPERPARAMETERS_TRAINING,
                                                                   HYPERPARAMETERS_MODEL_INIT)
from selecting_OOD_detector.utils.general import check_and_convert_dfs_to_numpy


def train_novelty_estimators(
        X_train: pd.DataFrame,
        y_train: pd.DataFrame = None,
        model_selection: set = None,
        hyperparameters_init: dict = HYPERPARAMETERS_MODEL_INIT,
        hyperparameters_training: dict = HYPERPARAMETERS_TRAINING,
        saved_model_dir: Optional[str] = None,

) -> dict:
    """
    A function to train available novelty estimators and return trained models in a dictionary.

    Parameters
        ----------
        X_train: pd.DataFrame
            Training data to fit novelty estimators on.
        y_train: Optional(pd.DataFrame)
            Labels corresponding to the training data (used only for for predictive models).
        model_selection: Optional(set)
            Define which models to train, e.g. {"PPCA", "LOF", "VAE"}
        hyperparameters_init: dict
            Hyperparameters used to initialize each model.
        hyperparameters_training: dict
            Hyperparameters used to train each model.
        saved_model_dir: Optional(str)
            If a path to saved models is provided, skips training and uses pre-trained model.
    """

    novelty_estimators = dict()

    # Check datasets and convert them to numpy arrays
    X_train, y_train = check_and_convert_dfs_to_numpy([X_train, y_train], allow_empty=True)

    if model_selection is None:
        model_selection = IMPLEMENTED_MODELS.keys()
    else:
        model_selection = model_selection & IMPLEMENTED_MODELS.keys()

    # Run or load each model
    for model_name in model_selection:

        # Try to load saved model
        failed_to_load_model = False
        if saved_model_dir is not None:
            try:
                ne = pickle.load(open(os.path.join(saved_model_dir, model_name), 'rb'))
                print(f"\tSuccesfully loaded {model_name}.")

            except FileNotFoundError as e:
                print(e,
                      f"\tCould not load {model_name}. Training instead...")
                failed_to_load_model = True

        # If a saved model is not available, train the model
        if saved_model_dir is None or failed_to_load_model:
            if not failed_to_load_model:
                print(f"\tTraining: {model_name}")

                init_params = hyperparameters_init[model_name]
                train_params = hyperparameters_training[model_name]

                init_params.update({"input_size": X_train.shape[1]})

                ne = IMPLEMENTED_MODELS[model_name](**init_params)
                ne.train(X_train, y_train=y_train, **train_params)

        novelty_estimators[model_name] = ne

    return novelty_estimators
