import abc
from enum import Enum
import logging
import numpy as np
import pandas as pd

try:
    import anndata
except ImportError:
    anndata = None

from .model import _Model_Base

logger = logging.getLogger(__name__)


class _Estimator_Base(_Model_Base, metaclass=abc.ABCMeta):
    r"""
    Estimator base class
    """

    class TrainingStrategy(Enum):
        AUTO = None

    def validate_data(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def initialize(self, **kwargs):
        """
        Initializes this estimator
        """
        pass

    @abc.abstractmethod
    def train(self, learning_rate=None, **kwargs):
        """
        Starts the training routine
        """
        pass

    def train_sequence(self, training_strategy=TrainingStrategy.AUTO):
        """
        Starts a sequence of training routines

        :param training_strategy: List of dicts or enum with parameters which will be passed to self.train().

                - `training_strategy = [ {"learning_rate": 0.5}, {"learning_rate": 0.05} ]` is equivalent to
                    `self.train(learning_rate=0.5); self.train(learning_rate=0.05);`

                - Can also be an enum: self.TrainingStrategy.[AUTO|DEFAULT|EXACT|QUICK|...]
                - Can also be a str: "[AUTO|DEFAULT|EXACT|QUICK|...]"
        """
        if isinstance(training_strategy, Enum):
            training_strategy = training_strategy.value
        elif isinstance(training_strategy, str):
            training_strategy = self.TrainingStrategy[training_strategy].value

        if training_strategy is None:
            training_strategy = self.TrainingStrategy.DEFAULT.value

        logger.info("training strategy: %s", str(training_strategy))

        for idx, d in enumerate(training_strategy):
            logger.info("Beginning with training sequence #%d", idx + 1)
            self.train(**d)
            logger.info("Training sequence #%d complete", idx + 1)

    @abc.abstractmethod
    def finalize(self, **kwargs):
        """
        Clean up, free resources

        :return: some Estimator containing all necessary data
        """
        pass

    @property
    @abc.abstractmethod
    def loss(self):
        pass

    @property
    @abc.abstractmethod
    def gradients(self):
        pass


class _EstimatorStore_XArray_Base:

    def __init__(self):
        pass

    def initialize(self, **kwargs):
        raise NotImplementedError("This object only stores estimated values")

    def train(self, **kwargs):
        raise NotImplementedError("This object only stores estimated values")

    def finalize(self, **kwargs):
        return self

    def validate_data(self, **kwargs):
        raise NotImplementedError("This object only stores estimated values")

    @property
    def input_data(self):
        return self._input_data

    @property
    def X(self):
        return self.input_data.X

    @property
    def features(self):
        return self.input_data.features

    @property
    def loss(self):
        return self.params["loss"]

    def _plot_coef_vs_ref(
            self,
            true_values: np.ndarray,
            estim_values: np.ndarray,
            size=1,
            log=False,
            save=None,
            show=True,
            ncols=5,
            row_gap=0.3,
            col_gap=0.25,
            title=None,
            return_axs=False
    ):
        """
        Plot estimated coefficients against reference (true) coefficients.

        :param true_values:
        :param estim_values:
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
        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        from matplotlib import rcParams

        plt.ioff()

        n_par = true_values.shape[0]
        ncols = ncols if n_par > ncols else n_par
        nrows = n_par // ncols + (n_par - (n_par // ncols) * ncols)

        gs = gridspec.GridSpec(
            nrows=nrows,
            ncols=ncols,
            hspace=row_gap,
            wspace=col_gap
        )

        fig = plt.figure(
            figsize=(
                ncols * rcParams['figure.figsize'][0],  # width in inches
                nrows * rcParams['figure.figsize'][1] * (1 + row_gap)  # height in inches
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
            y = estim_values[i, :]
            if log:
                x = np.log(x + 1)
                y = np.log(y + 1)

            sns.scatterplot(
                x=x,
                y=y,
                size=size,
                ax=ax,
                legend=False
            )
            sns.lineplot(
                x=np.array([np.min([np.min(x), np.min(y)]), np.max([np.max(x), np.max(y)])]),
                y=np.array([np.min([np.min(x), np.min(y)]), np.max([np.max(x), np.max(y)])]),
                ax=ax
            )

            title_i = title + "_" + str(i)
            # Add correlation into title:
            title_i = title_i + " (R=" + str(np.round(np.corrcoef(x, y)[0, 1], 3)) + ")"
            ax.set_title(title_i)
            ax.set_xlabel("true parameter")
            ax.set_ylabel("estimated parameter")

        # Save, show and return figure.
        if save is not None:
            plt.savefig(save + '_parameter_scatter.png')

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if return_axs:
            return axs
        else:
            return

    def _plot_deviation(
            self,
            true_values: np.ndarray,
            estim_values: np.ndarray,
            save=None,
            show=True,
            title=None,
            return_axs=False
    ):
        """
        Plot estimated coefficients against reference (true) coefficients.

        :param true_values:
        :param estim_values:
        :param save: Path+file name stem to save plots to.
            File will be save+"_genes.png". Does not save if save is None.
        :param show: Whether to display plot.
        :param return_axs: Whether to return axis objects.
        :return: Matplotlib axis objects.
        """
        import seaborn as sns
        import matplotlib.pyplot as plt

        plt.ioff()

        n_par = true_values.shape[0]
        summary_fit = pd.concat([
            pd.DataFrame({
                "deviation": estim_values[i, :] - true_values[i, :],
                "coefficient": pd.Series(["coef_"+str(i) for x in range(estim_values.shape[1])], dtype="category")
            }) for i in range(n_par)])
        summary_fit['coefficient'] = summary_fit['coefficient'].astype("category")

        fig, ax = plt.subplots()
        sns.violinplot(
            x=summary_fit["coefficient"],
            y=summary_fit["deviation"],
            ax=ax
        )

        if title is not None:
            ax.set_title(title)

        # Save, show and return figure.
        if save is not None:
            plt.savefig(save + '_deviation_violin.png')

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if return_axs:
            return ax
        else:
            return

