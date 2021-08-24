from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statannot import add_stat_annotation

plt.style.use("default")
plt.rcParams.update(plt.rcParamsDefault)


def plot_scores_boxplot(scores_test: Union[dict, pd.DataFrame],
                        scores_new: Union[dict, pd.DataFrame],
                        show_outliers: bool = True,
                        stat_test: str = "Mann-Whitney-ls",
                        return_results: bool = True,
                        title: str = "",
                        figsize=(15, 8),
                        save_dir: Optional[str] = None,
                        ):
    """
    Plots scores of two datasets as a boxplot with statistical annotation.
    Parameters
    ----------
    scores_test:  Union[dict, pd.DataFrame]
        Test scores (baseline) obtained by each model.
    scores_new:  Union[dict, pd.DataFrame]
        New scores to be compared with the test scores.
    show_outliers: bool
        Indicates whether outliers are shown in the plot.
    stat_test: str
        Specify which statistical test should be used for comparison. Must be one of: [`Levene`, `Mann-Whitney`,
        `Mann-Whitney-gt`, `Mann-Whitney-ls`,  `t-test_ind`, `t-test_welch`, `t-test_paired`, `Wilcoxon`, `Kruskal`]
    return_results: bool
        If true, returns a dictionary of StatResult objects for each model.
    title : str
        Title to appear on the plot
    figsize

    Returns
    -------
    test_results: Optional(dict[StatResult])
        Dictionary of StatResult objects for each model that contain a desired statistic and a p-value.


    """
    plt.style.use("default")
    assert set(scores_test.keys()) == set(scores_new.keys())

    fig = plt.figure(figsize=figsize)

    columns = scores_test.keys()

    df_test, df_new = pd.DataFrame(scores_test), pd.DataFrame(scores_new)
    df_test["type"] = ["Test"] * df_test.shape[0]
    df_new["type"] = ["OOD"] * df_new.shape[0]
    df_scores = pd.concat([df_test, df_new])

    if return_results:
        test_results = {}

    for i, key in enumerate(columns):
        ax = fig.add_subplot(3, 6, i + 1)

        sns.boxplot(data=df_scores,
                    x="type",
                    y=key,
                    ax=ax,
                    showfliers=show_outliers)

        ax.set_xticklabels(["Test Data", "OOD Data"])
        ax.set_title(key)

        stats = add_stat_annotation(ax,
                                    data=df_scores,
                                    x="type",
                                    y=key,
                                    box_pairs=[("Test", "OOD")],
                                    test=stat_test,
                                    loc='outside',
                                    verbose=0,
                                    line_offset_to_box=0.2)
        ax.set_xlabel("")
        ax.set_ylabel("Novelty Scores")

        if return_results:
            test_results[key] = stats

    plt.suptitle(title, y=1.005, x=0.45)
    fig.tight_layout(pad=1.0)

    if save_dir is not None:
        plt.savefig(save_dir, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    if return_results:
        return test_results


def plot_scores_distr(scores_test: Union[dict, pd.DataFrame],
                      scores_new: Union[dict, pd.DataFrame],
                      clip_q: float = 0,
                      save_dir: Optional[str] = None,
                      figsize=(15, 6),
                      title="",
                      labels=None,
                      **kwargs,
                      ):
    """

    Parameters
    ----------
    save_dir
    scores_test:  Union[dict, pd.DataFrame]
        Test scores (baseline) obtained by each model.
    scores_new:  Union[dict, pd.DataFrame]
        New scores to be compared with the test scores.
    clip_q: float
        Float that specifies the inter-quantile region to be plotted. If zero, all values are plotted.
        It is used to remove outlier points from the plot to better see details of the distrbutions.
        Example: clip_q = 0.05 results in plotting the interval of 0.05% scores - 0.95 % new scores.
    figsize
    **kwargs
        Arguments passed to the pandas.DataFrame.plot() function

    """
    plt.style.use("default")
    assert 0 <= clip_q < 0.5
    assert set(scores_test.keys()) == set(scores_new.keys())

    fig = plt.figure(figsize=figsize)

    for i, key in enumerate(scores_test.keys()):
        scores_test_ax = pd.Series(scores_test[key])
        scores_new_ax = pd.Series(scores_new[key])
        ax = fig.add_subplot(3, 6, i + 1)

        clip_min = min(min(scores_test_ax), min(scores_new_ax))
        clip_max = max(scores_test_ax.quantile(1 - clip_q), scores_new_ax.quantile(1 - clip_q))

        np.clip(scores_test_ax, clip_min, clip_max).plot(ax=ax,
                                                         label="Test",
                                                         alpha=0.9,
                                                         density=True,
                                                         **kwargs)

        np.clip(scores_new_ax, clip_min, clip_max).plot(ax=ax,
                                                        label="OOD",
                                                        density=True,
                                                        alpha=0.6,
                                                        **kwargs)
        label = "Novelty Score"
        if labels is not None:
            try:
                label = labels[key]
            except ValueError:
                pass

        ax.set_xlabel(label)
        ax.legend()
        ax.set_title(key)

    fig.tight_layout(pad=1.0)
    plt.suptitle(title, y=1.01)

    if save_dir is not None:
        plt.savefig(save_dir, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_heatmap(df: pd.DataFrame,
                 title: str = "",
                 figsize: tuple = (10, 3),
                 save_dir: Optional[str] = None,
                 annot: [bool, np.ndarray] = True,
                 vmin: float = 0.5,
                 vmax: float = 1.0,
                 cmap: str = "OrRd",
                 **kwargs):
    """
    Plots provided dataframe as sns.heatmap.
    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to be plotted.
    title: str
        Title to display on the top of the plot.
    figsize: tuple
        Indicates size of the image.
    save_dir: Optional(str)
        If provided, the plot will be saved in this directory.
    annot: bool, np.ndarray
        If annot is set to True, adds a simple annotation of the value to the plotted heatmap. Alternatively,
        a custom annotation in an array can be provided where array shape corresponds to the dataframe shape.
    vmin, vmax : float
        Minimal and maximal value to use in the colorbar.
    cmap: str
        Color map to be used.
    kwargs:
        Other arguments to be passed to sns.heatmap
    Returns
    -------

    """
    plt.style.use("default")
    _, ax = plt.subplots(figsize=figsize)
    sns.set(font_scale=0.9)
    sns.heatmap(
        df.T,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cmap=cmap,
        annot=annot,
        **kwargs
    )
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")

    plt.xticks()
    plt.yticks()

    plt.title(title)

    if save_dir is not None:
        plt.savefig(save_dir, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
