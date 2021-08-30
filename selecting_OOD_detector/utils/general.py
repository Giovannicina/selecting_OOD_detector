import os
import json
import pandas as pd
import numpy as np


def check_and_convert_dfs_to_numpy(dfs, allow_empty=True):
    """
    Converts dataframes to numpy arrays. Implemented to unify conversion if one of the arrays is None (e.g. if labels
    are not provided) or provided as a numpy array.

    Parameters
    ----------
    dfs: set or list of pd.DataFrames or array-like objects
        Dataframes to be converted.
    allow_empty: bool
        Indicates whether None is allowed in the dataframes. If yes, returns None in place of the dataframe.
    Returns
    -------
    list of np.ndarrays
        Returns a list of numpy arrays.
    """
    arrays = []

    for df in dfs:

        if df is None:
            if allow_empty:
                arrays.append(df)
            else:
                raise ValueError("Encountered an empty dataframe.")

        elif type(df) is pd.DataFrame or type(df) is pd.Series:
            arrays.append(df.values)

        elif type(df) is np.ndarray:
            arrays.append(df)

        else:
            raise ValueError(f"Unknown data type provided: {type(df)}.")

    return arrays


def save_dictionary_as_json(dictn, save_name, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(f"{save_dir}/{save_name}.json", "w") as result_file:
        result_file.write(json.dumps(dictn, indent=4, default=str))
