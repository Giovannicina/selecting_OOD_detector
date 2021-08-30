from collections import defaultdict

import pandas as pd
import numpy as np

from selecting_OOD_detector.utils.general import check_and_convert_dfs_to_numpy
from sklearn.metrics import roc_auc_score


def score_dataset(X: pd.DataFrame,
                  models_trials_dict: dict,
                  ) -> dict:
    """
    Parameters
    ----------

    X: pd.DataFrame
        Dataset to be scored.
    models_trials_dict: dict
         Nested dictionary of novelty estimators. Contains 1 or more trials.
         Example of a dictionary containing 2 trials:
                  {"0": {"AE": NoveltyEstimator, "VAE": NoveltyEstimator},
                  "1": {"AE": NoveltyEstimator, "VAE": NoveltyEstimator}}
    Returns
    -------
    scores: dict
        Dictionary of the novelty scores obtained by each model.
    """
    X = check_and_convert_dfs_to_numpy([X])[0]

    scores = dict()

    for model_name in list(models_trials_dict.values())[0]:
        scores[model_name] = np.array([
            models_trials_dict[i][model_name].get_novelty_score(X)
            for i in range(len(models_trials_dict))
        ])

    return scores


def get_ood_aucs_score_for_all_models(ood_scores_trials_dict: dict,
                                      test_scores_trials_dict: dict):
    """

    Parameters
    ----------
    ood_scores_trials_dict: dict
        A nested dictionary that contains: a name of OOD group and names of novelty estimators and the scores.
    test_scores_trials_dict
        A dictionary that contains names of novelty estimators and the scores.
    Returns
    -------

    """
    aucs_dict = defaultdict(list)

    for model_name, ood_scores_models in ood_scores_trials_dict.items():
        for trial in range(len(ood_scores_models)):

            auc_score = _get_ood_aucs_score_(ood_scores_models[trial],
                                             test_scores_trials_dict[model_name][trial])
            aucs_dict[model_name].append(auc_score)

    return aucs_dict


def _get_ood_aucs_score_(ood_uncertainties: np.ndarray,
                         test_uncertainties: np.ndarray) -> float:
    """
    Return AUC-ROC score of OOD detection.
    """

    all_uncertainties = np.concatenate([ood_uncertainties,
                                        test_uncertainties])

    labels = np.concatenate([np.ones(len(ood_uncertainties)),
                             np.zeros(len(test_uncertainties))])

    return roc_auc_score(labels, all_uncertainties)


def average_values_in_nested_dict(nested_dict):
    """
    Returns a dictionary with averaged inner values.
    Parameters
    ----------
    nested_dict: dict
        A dictionary of the following structure: {"Outer Key 1": {"Inner Key 1": [0.5, 0.6, 0.7]}}

    Returns
    -------
    dict
        A dictionary with an averaged value of the inner lists {"Outer Key 1": {"Inner Key 1": [0.6}}
    """
    averaged_nested_df = pd.DataFrame(nested_dict).applymap(lambda x: np.asarray(x).mean())

    return averaged_nested_df.to_dict()
