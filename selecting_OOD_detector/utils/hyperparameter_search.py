"""
A module with helper functions for running a hyperparameter search.
Code adopted form https://github.com/Pacmed/ehr_ood_detection/blob/master/src/experiments/hyperparameter_search.py
"""
from typing import Optional
import os

from sklearn.model_selection import ParameterSampler
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from selecting_OOD_detector.utils.general import check_and_convert_dfs_to_numpy, save_dictionary_as_json
from selecting_OOD_detector.models.novelty_estimator import NoveltyEstimator


def sample_hyperparameters(
        model_name: str,
        hyperparameter_grid: dict,
        hyperparameters_names: dict,
        n_evals: int = 20,
):
    """
    Sample the hyperparameters for different runs of the same model. The distributions parameters are sampled from are
    defined the provided hyperparamter grid.

    Parameters
    ----------

    model_name: str
        Name of the model.
    hyperparameter_grid: dict
        Dictionary of all possible values to be tested.
    hyperparameters_names: dict
        Dictionary containing model names and names of hyperparamaters that they use.
    n_evals: int
        Number of evaluations to run for the model.

    Returns
    -------
    sampled_params: list
        List of dictionaries containing hyperparameters and their sampled values.
    """
    sampled_params = list(
        ParameterSampler(
            param_distributions={
                hyperparam: hyperparameter_grid[hyperparam]
                for hyperparam in hyperparameters_names[model_name]
                if hyperparam in hyperparameter_grid
            },
            n_iter=n_evals,
        )
    )

    return sampled_params


def evaluate_set_of_parameters(model: NoveltyEstimator,
                               X_train: pd.DataFrame,
                               X_val: pd.DataFrame,
                               train_params: dict,
                               y_train: Optional[pd.DataFrame] = None,
                               y_val: Optional[pd.DataFrame] = None):
    """
    Runs a single round of training and evaluation for a set of paramaters.
    Parameters
    ----------
    model: NoveltyEstimator
        Model to be trained and evaluated.
    X_train: pd.DataFrame
        Training data.
    X_val: pd.DataFrame
        Validation data to calculate scores on.
    train_params: dict
        Parameters to be added to  ``train`` function of the model.
    y_train: Optional(pd.DataFrame):
         Labels corresponding to the training data. Only used for discriminator models.
    y_val: Optional(pd.DataFrame)
        Labels corresponding to the validation data. Only used for discriminator models.

    Returns
    -------
    score: float
        Score corresponding to the performance of the model. Either AUC-ROC score of predicting the correct
        labels for discriminators or likelihood of data for density estimators.

    """
    X_train, X_val, y_train, y_val = check_and_convert_dfs_to_numpy([X_train, X_val, y_train, y_val])

    model.train(X_train, y_train=y_train, **train_params)

    # For density estimators, evaluate according to the highest likelihood on data (same as the lowest novelty score)
    if model.model_type == "density_estimator":
        preds = -model.get_novelty_score(X_val)
        score = float(preds.mean())

    # For discriminators, evaluate according to the lowest prediction error using AUC-ROC score
    elif model.model_type == "discriminator":
        preds = model.predict_proba(X_val)
        if np.isnan(preds).all():
            score = 0

        else:
            preds = preds[:, 1]
            score = roc_auc_score(
                y_true=y_val[~np.isnan(preds)],
                y_score=preds[~np.isnan(preds)],
            )
            print(f"\tscore: {score}")
    else:
        raise NotImplementedError("Only density estimators and discriminators are implemented at the moment.")

    return score


def evaluate_hyperparameters(model_name: str,
                             model_class: NoveltyEstimator,
                             X_train: pd.DataFrame,
                             X_val: pd.DataFrame,
                             hyperparameter_grid: dict,
                             hyperparameters_names: dict,
                             train_params: dict,
                             y_train: pd.DataFrame = None,
                             y_val: pd.DataFrame = None,
                             num_evals: int = 20,
                             save_intermediate_scores: bool = True,
                             save_dir: Optional[str] = None,
                             ):

    scores, sorted_scores = {}, {}
    sampled_params = sample_hyperparameters(model_name,
                                            hyperparameter_grid=hyperparameter_grid,
                                            hyperparameters_names=hyperparameters_names,
                                            n_evals=num_evals)

    for run, param_set in enumerate(sampled_params):
        print(f"\t{run + 1}/{len(sampled_params)}", end=" ")
        param_set.update(input_size=X_train.shape[1])

        model = model_class(**param_set)

        # Run a single evaluation on the set of parameters
        try:
            score = evaluate_set_of_parameters(model=model, train_params=train_params,
                                               X_train=X_train, X_val=X_val,
                                               y_train=y_train, y_val=y_val)

        # In case of nans due bad training parameter
        except (ValueError, RuntimeError) as e:
            print(f"\tskipped the current run due to an error: {str(e)}", end=" ")
            score = -np.inf

        if np.isnan(score):
            score = -np.inf

        # Save results of the single run
        print(f"\tscore = {round(score, 2)}")
        scores[run] = {"score": score, "hyperparameters": param_set}

        # Sort the scores such that the best performing paramameters are displayed first
        sorted_scores = dict(
            list(sorted(scores.items(), key=lambda run: run[1]["score"], reverse=True))
        )
        # Save results for each run in case of an unexpected interruption
        if save_intermediate_scores:
            _save_hyperparameter_scores(scores=sorted_scores, model_name=model_name, save_dir=save_dir)

    return sorted_scores


def _save_hyperparameter_scores(scores, model_name, save_dir=None):
    """
    Saves scores and parameters for a model to a json file.
    """
    if save_dir is None:
        save_dir = "../data/hyperparameters/scores/"

    save_dictionary_as_json(dictn=scores, save_name=f"scores_{model_name}", save_dir=save_dir)

