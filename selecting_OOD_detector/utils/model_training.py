import os
from typing import Optional
import pickle

import pandas as pd

from selecting_OOD_detector.models.novelty_estimators_info import (IMPLEMENTED_MODELS,
                                                                   HYPERPARAMETERS_TRAINING,
                                                                   HYPERPARAMETERS_MODEL_INIT)
from selecting_OOD_detector.utils.general import check_and_convert_dfs_to_numpy


def load_novelty_estimator(
        model_name: str,
        saved_model_dir: str = None,
):
    """

    Parameters
    ----------
    model_name: str
        Indicates which model to load. E.g. "AE" or "PPCA"
    saved_model_dir: str
        Path to the directory with saved models.

    Returns
    -------
        NoveltyEstimator or None
        If model was loaded, returns NoveltyEstimator object. Else returns None

    """
    try:
        ne = pickle.load(open(os.path.join(saved_model_dir, model_name), 'rb'))
        print(f"\t\tsuccesfully loaded {model_name}.")
        return ne

    except FileNotFoundError as e:
        print(e,
              f"\tcould not load {model_name}.")
        return None


def train_novelty_estimator(
        model_name: str,
        X_train: pd.DataFrame,
        init_params: dict,
        train_params: dict,
        y_train: pd.DataFrame = None,
):
    """

    Parameters
    ----------
    model_name: str
        Indicates which model to load. E.g. "AE" or "PPCA"
    X_train: pd.DataFrame
        Training data to fit novelty estimators on.
    init_params: dict
        Hyperparameters used to initialize the model.
    train_params: dict
        Hyperparameters used to train the model.
    y_train: pd.DataFrame
        Labels corresponding to the training data (used only for for predictive models).
    Returns
    -------

    """
    print(f"\t\t{model_name}...", end=" ")
    init_params.update({"input_size": X_train.shape[1]})
    ne = IMPLEMENTED_MODELS[model_name](**init_params)
    ne.train(X_train, y_train=y_train, **train_params)
    print(f"done.")
    return ne


def get_novelty_estimators(
        X_train: pd.DataFrame,
        y_train: pd.DataFrame = None,
        model_selection: set = None,
        hyperparameters_init: dict = HYPERPARAMETERS_MODEL_INIT,
        hyperparameters_training: dict = HYPERPARAMETERS_TRAINING,
        saved_model_dir: Optional[str] = None
):
    """
    A function to train available novelty estimators and return trained models in a dictionary.

    Parameters
        ----------
        X_train: pd.DataFrame
            Training data to fit novelty estimators on.
        y_train: Optional(pd.DataFrame)
            Labels corresponding to the training data (used only for for predictive models).
        model_selection: Optional(set)
            Define which models to train, e.g. {"PPCA", "LOF", "VAE"}. If selection is not provided, all available
            models are used.
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

        ne = None

        # Try to load saved model
        if saved_model_dir is not None:
            ne = load_novelty_estimator(model_name=model_name, saved_model_dir=saved_model_dir)

        # If no directory with saved models is provided or loading model was not successful, train it
        if saved_model_dir is None or ne is None:
            init_params = hyperparameters_init[model_name]
            train_params = hyperparameters_training[model_name]

            ne = train_novelty_estimator(X_train=X_train,
                                         y_train=y_train,
                                         model_name=model_name,
                                         init_params=init_params,
                                         train_params=train_params)

        novelty_estimators[model_name] = ne

    return novelty_estimators
