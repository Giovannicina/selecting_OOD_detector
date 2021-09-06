"""
This module implements HyperparameterTuner which performs hyperparameter search for density estimators
and discriminators.
"""
from typing import Optional
from collections import defaultdict

import pandas as pd

from selecting_OOD_detector.utils.general import save_dictionary_as_json
from selecting_OOD_detector.models.novelty_estimators_info import (
    HYPERPARAMETERS_TRAINING,
    HYPERPARAMETERS_MODEL_INIT,
    HYPERPARAMETERS_SEARCH_GRID,
    IMPLEMENTED_MODELS
)
from selecting_OOD_detector.utils.hyperparameter_search import (
    evaluate_hyperparameters
)


class HyperparameterTuner:
    """
    Performs hyperparameter search for implemented novelty estimators and provided data.
    """

    def __init__(self,
                 hyperparameter_search_grid: Optional[dict] = None,
                 train_params: dict = None,
                 model_selection: Optional[set] = None,
                 num_evals_per_model: int = 20,
                 ):
        """
        Parameters
        ----------
        hyperparameter_search_grid: dict
            A dictionary that specifies all possible values of hyperparameters for all models to be evaluated.
            Example:   {"kernel": ["RFB", "Matern12", "Matern32", "Matern52", "RQ"],
                       "n_inducing_points": range(10, 20)}
        train_params: dict
            A dictionary that specifies hyperparameters used in the training function.
        model_selection: Optional(set)
            Specifies a subset of models to be evaluated. If no selectio is provided, evaluates all implemented models.
        num_evals_per_model: int
            Number of evaluations to run for each model.
        """
        # Initialize hyperparameter search grid
        if hyperparameter_search_grid is None:
            self.hyperparameter_search_grid = HYPERPARAMETERS_SEARCH_GRID
        else:
            self.hyperparameter_search_grid = hyperparameter_search_grid

        # Initialize a dictionary with hyperparameter names for each model used to
        # distinguish which model uses which hyperparameters for initialization and training
        self.hyperparameters_names = HYPERPARAMETERS_MODEL_INIT

        # Initialize training hyperparameters
        if train_params is None:
            self.train_params = HYPERPARAMETERS_TRAINING
        else:
            self.train_params = train_params

        # Specify which models are to be trained
        if model_selection is None:
            self.model_selection = IMPLEMENTED_MODELS.keys()
        else:
            assert [model_name in IMPLEMENTED_MODELS.keys() for model_name in model_selection], \
                f"Invalid model selection. Please select models from {IMPLEMENTED_MODELS.keys()}."
            self.model_selection = model_selection

        self.num_evals_per_model = num_evals_per_model
        self.evaluated_parameters = defaultdict(dict)

    def run_hyperparameter_search(self,
                                  X_train: pd.DataFrame,
                                  X_val: pd.DataFrame,
                                  y_train: pd.DataFrame = None,
                                  y_val: pd.DataFrame = None,
                                  save_intermediate_scores: bool = True,
                                  save_dir: Optional[str] = None,
                                  ):
        """
        Performs hyperparameters search for all models and stores the results internally.
        Parameters
        ----------
        X_train: pd.DataFrame
            Training data.
        X_val: pd.DataFrame
            Validation data to calculate scores on.
        y_train: Optional(pd.DataFrame):
            Labels corresponding to the training data. Only used for discriminator models.
        y_val: Optional(pd.DataFrame)
            Labels corresponding to the validation data. Only used for discriminator models.
        save_intermediate_scores: bool
            If True, saves results after each run in case of an abrupt termination of the run.
        save_dir: Optional(str)
            If save_intermediate_scores is True, saves the results to provided directory. If no directory is provided,
            saves the results to "../data/hyperparameters/scores/")
        """

        for model_name in self.model_selection:
            print(f"Model: {model_name}")
            model_class = IMPLEMENTED_MODELS[model_name]

            sorted_scores = evaluate_hyperparameters(
                model_name=model_name,
                model_class=model_class,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                hyperparameter_grid=self.hyperparameter_search_grid,
                hyperparameters_names=self.hyperparameters_names,
                train_params=self.train_params[model_name],
                num_evals=self.num_evals_per_model,
                save_intermediate_scores=save_intermediate_scores,
                save_dir=save_dir,
            )

            self.evaluated_parameters[model_name] = sorted_scores

    def get_best_parameteres(self):
        """
        Returns the top performing paramaters for each model in a nested dictionary.
        Returns
        -------
        best_params: dict
            A nested dictionary that stores the best parameters found for each model.
        """
        best_params = {}

        for model_name in self.evaluated_parameters.keys():
            best_run_number = list(self.evaluated_parameters[model_name].keys())[0]
            best_model_params = self.evaluated_parameters[model_name][best_run_number]

            best_params[model_name] = best_model_params["hyperparameters"]

        return best_params

    def save_best_parameters_as_json(self, save_dir: Optional[str] = None):
        """
        Saves the top performing paramaters for each model in a nested dictionary.
        Parameters
        ----------
        save_dir: Optional(str)
            Directory to save the results to. If no directory is provided, saves the paramaters to:
            "../data/hyperparameters/custom/"

        """
        init_params = self.get_best_parameteres()
        train_params = self.train_params

        if save_dir is None:
            save_dir = "../data/hyperparameters/custom/"
            print(f"Saving results to the following directory: {save_dir}.")

        save_dictionary_as_json(dictn=init_params, save_name="init", save_dir=save_dir)
        save_dictionary_as_json(dictn=train_params, save_name="train", save_dir=save_dir)

    def save_scores_for_evaluated_paramaters(self, save_dir: Optional[str] = None):
        """
        Saves scores for all evaluated parameters for each model.
        Parameters
        ----------
        save_dir: Optional(str)
            Directory to save the results to. If no directory is provided, saves the paramaters to:
            "../data/hyperparameters/scores/")
        """
        if save_dir is None:
            save_dir = "../data/hyperparameters/scores/"
            print(f"Saving results to the following directory: {save_dir}.")

        save_dictionary_as_json(dictn=self.evaluated_parameters, save_name=f"scores_all_models", save_dir=save_dir)
