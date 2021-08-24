from collections import defaultdict
from typing import Optional

import pandas as pd
import numpy as np
from astropy.stats import jackknife_stats

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


def average_values_in_nested_dict(nested_dict,
                                  axis: Optional[int] = None,
                                  dict_level: int = 2) -> dict:
    """
    Takes in a dictionary with lists and returns a dictionary with averaged values of those lists.

    Parameters
    ----------
    nested_dict: dict
        A nested dictionary with two levels (e.g.: {"Outer Key 1": {"Inner Key 1": [0.5, 0.6, 0.7]}}).
        If the dictionary to be averaged has only one level, (e.g.: {"Key": [0.5, 0.6, 0.7]}) use dict_level = 1.
    axis: int
        Axis to apply the mean fuction along. If none is provided, all values are averaged.
    dict_level: int
        If level is set to 2, nested dictionary is expected. If level is 1, a standard dictionary is expected.

    Returns
    -------
    dict
        A dictionary with an averaged value of the inner lists {"Outer Key 1": {"Inner Key 1": [0.6}}. If level==1,
        simply averages the values under each key in the dictionary {"Outer Key": [0.6]}.
    """
    assert dict_level == 2 or dict_level == 1, " Please provide a valid level number (1 or 2). " \
                                               "For nested dictonary set dict_level=2, for standard" \
                                               "dictionary set dict_level=1."
    # Dictionary with depth level 1
    if dict_level == 1:
        averaged_nested_df = pd.Series(nested_dict).apply(lambda x: np.asarray(x).mean(axis=axis))

    # Dictionary with depth level 2
    else:
        averaged_nested_df = pd.DataFrame(nested_dict).applymap(lambda x: np.asarray(x).mean(axis=axis))

    return averaged_nested_df.to_dict()


def get_mean_stderr_annots_in_nested_dict(nested_dict,
                                          as_string=True,
                                          dict_level: int = 2) -> dict:
    """
    Returns means and standard errors of the values inside a nested dictionary.

    Parameters
    ----------

    nested_dict: dict
       A dictionary of the following structure: {"Outer Key 1": {"Inner Key 1": [0.5, 0.6, 0.7]}.
        If the dictionary to be averaged has only one level, (e.g.: {"Key": [0.5, 0.6, 0.7]}) use dict_level = 1.
    as_string: bool
        If True, returns the result in strings: "mean ± stderr". Else, returns lists: [mean, stderr].
    dict_level: int
        If level is set to 2, nested dictionary is expected. If level is 1, a standard dictionary is expected.

    Returns
    -------
    means_errors: dict
        A dictionary with a string or lists of means and standard errors.
         Example: {"Outer Key 1": {"Inner Key 1": ["0.6 ± 0.1"]}}

    """
    assert dict_level == 2 or dict_level == 1, " Please provide a valid level number (1 or 2). " \
                                               "For nested dictonary set dict_level=2, for standard" \
                                               "dictionary set dict_level=1."

    means_errors = defaultdict(lambda: defaultdict(list))

    # Dictionary with depth level 1
    if dict_level == 1:
        for inner_key, inner_value in nested_dict.items():
            mean, _, stderr, _ = jackknife_stats(np.array(inner_value), np.mean)

            if as_string:
                means_errors[inner_key] = f"{np.round(mean, 3)} ± {np.round(stderr, 3)}"

            else:
                means_errors[inner_key] = [mean, stderr]

    #  Dictionary with depth level 2
    else:
        for outter_key, outter_value in nested_dict.items():
            for inner_key, inner_value in outter_value.items():
                mean, _, stderr, _ = jackknife_stats(np.array(inner_value), np.mean)

                if as_string:
                    means_errors[outter_key][inner_key] = f"{np.round(mean, 3)} ± {np.round(stderr, 3)}"

                else:
                    means_errors[outter_key][inner_key] = [mean, stderr]

    return means_errors
