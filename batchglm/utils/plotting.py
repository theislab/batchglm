import logging
from typing import Optional, Tuple, Union

import dask.array
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec, rcParams
from matplotlib.axes import Axes

logger = logging.getLogger(__name__)


def _input_checks(
    true_values: Union[np.ndarray, dask.array.core.Array], pred_values: Union[np.ndarray, dask.array.core.Array]
):
    """
    Check the type of true and predicted input and make sure they have the same size.

    :param true_values: The reference parameters.
    :param pred_values: The fitted parameters.
    """

    def _cast(data: Union[np.ndarray, dask.array.core.Array]) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(data, dask.array.core.Array):
            to_return = data.coompute()
        elif isinstance(data, np.ndarray):
            to_return = data
        else:
            raise TypeError(f"Type {type(data)} is not recognized for true/pred values.")
        return to_return

    true_vals = _cast(true_values)
    pred_vals = _cast(pred_values)

    assert len(true_values.shape) == len(pred_values.shape), "true_values must have same dimensions as pred_values"
    assert np.all(true_values.shape == pred_values.shape), "true_values must have same dimensions as pred_values"

    return true_vals, pred_vals


def plot_coef_vs_ref(
    true_values: Union[np.ndarray, dask.array.core.Array],
    pred_values: Union[np.ndarray, dask.array.core.Array],
    size=1,
    log=False,
    save=None,
    show=True,
    ncols=5,
    row_gap=0.3,
    col_gap=0.25,
    title: str = "",
    return_axs: bool = False,
) -> Optional[Axes]:
    """
    Plot estimated coefficients against reference (true) coefficients for location model.

    :param true_values:
    :param size: Point size.
    :param save: Path+file name stem to save plots to.
        File will be save+"_genes.png". Does not save if save is None.
    :param show: Whether to display plot.
    :param ncols: Number of columns in plot grid if multiple genes are plotted.
    :param row_gap: Vertical gap between panel rows relative to panel height.
    :param col_gap: Horizontal gap between panel columns relative to panel width.
    :param title: Plot title.
    :param return_axs: Whether to return axis objects.
    :return: Matplotlib axis objects.
    """
    true_values, pred_values = _input_checks(true_values, pred_values)

    plt.ioff()

    n_par = true_values.shape[0]
    ncols = ncols if n_par > ncols else n_par
    nrows = n_par // ncols + (n_par - (n_par // ncols) * ncols)

    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, hspace=row_gap, wspace=col_gap)

    fig = plt.figure(
        figsize=(
            ncols * rcParams["figure.figsize"][0],  # width in inches
            nrows * rcParams["figure.figsize"][1] * (1 + row_gap),  # height in inches
        )
    )

    if title is None:
        title = "parameter"

    # Build axis objects in loop.
    axs = []
    for i in range(n_par):
        ax = plt.subplot(gs[i])
        axs.append(ax)

        x = true_values[i, :]
        y = pred_values[i, :]
        if log:
            x = np.log(x + 1)
            y = np.log(y + 1)

        sns.scatterplot(x=x, y=y, size=size, ax=ax, legend=False)
        sns.lineplot(
            x=np.array([np.min([np.min(x), np.min(y)]), np.max([np.max(x), np.max(y)])]),
            y=np.array([np.min([np.min(x), np.min(y)]), np.max([np.max(x), np.max(y)])]),
            ax=ax,
        )

        title_i = title + "_" + str(i)
        # Add correlation into title:
        title_i = title_i + " (R=" + str(np.round(np.corrcoef(x, y)[0, 1], 3)) + ")"
        ax.set_title(title_i)
        ax.set_xlabel("true parameter")
        ax.set_ylabel("estimated parameter")

    # Save, show and return figure.
    if save is not None:
        plt.savefig(save + "_parameter_scatter.png")

    if show:
        plt.show()

    plt.close(fig)
    plt.ion()

    if return_axs:
        return axs
    return None


def plot_deviation(
    true_values: np.ndarray, pred_values: np.ndarray, save=None, show=True, return_axs=False, title: str = ""
) -> Optional[Axes]:
    """
    Plot deviation of estimated coefficients from reference (true) coefficients
    as violin plot for location model.

    :param true_values:
    :param pred_values:
    :param save: Path+file name stem to save plots to.
        File will be save+"_genes.png". Does not save if save is None.
    :param show: Whether to display plot.
    :param return_axs: Whether to return axis objects.
    :param title: Title.
    :return: Matplotlib axis objects.
    """
    true_values, pred_values = _input_checks(true_values, pred_values)

    plt.ioff()

    n_par = true_values.shape[0]
    summary_fit = pd.concat(
        [
            pd.DataFrame(
                {
                    "deviation": pred_values[i, :] - true_values[i, :],
                    "coefficient": pd.Series(["coef_" + str(i) for x in range(pred_values.shape[1])], dtype="category"),
                }
            )
            for i in range(n_par)
        ]
    )
    summary_fit["coefficient"] = summary_fit["coefficient"].astype("category")

    fig, ax = plt.subplots()
    sns.violinplot(x=summary_fit["coefficient"], y=summary_fit["deviation"], ax=ax)

    if title is not None:
        ax.set_title(title)

    # Save, show and return figure.
    if save is not None:
        plt.savefig(save + "_deviation_violin.png")

    if show:
        plt.show()

    plt.close(fig)
    plt.ion()

    if return_axs:
        return ax
    return None
