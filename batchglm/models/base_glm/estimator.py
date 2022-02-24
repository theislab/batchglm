import abc
import logging
import pprint
import sys
from enum import Enum
from typing import Union

import dask
import numpy as np
import pandas as pd

try:
    import anndata
except ImportError:
    anndata = None

from .model import _ModelGLM

logger = logging.getLogger(__name__)


class _EstimatorGLM(metaclass=abc.ABCMeta):
    """
    Estimator base class for generalized linear models (GLMs).
    Adds plotting functionality.

    Attributes
    ----------
    model : _ModelGLM
        Model to fit
    """

    model: _ModelGLM

    def __init__(self, model: _ModelGLM):
        """
        Create a new _EstimatorGLM object.

        :param data: Some data object.
        :param model: Model to fit
        """
        self.model = model
        self._loss = None
        self._log_likelihood = None
        self._jacobian = None
        self._hessian = None
        self._fisher_inv = None

    def plot_coef_location_vs_ref(
        self,
        true_values: np.ndarray,
        size=1,
        log=False,
        save=None,
        show=True,
        ncols=5,
        row_gap=0.3,
        col_gap=0.25,
        return_axs=False,
    ):
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
        :param return_axs: Whether to return axis objects.
        :return: Matplotlib axis objects.
        """
        assert len(true_values.shape) == len(
            self.model.theta_location.shape
        ), "true_values must have same dimensions as self.theta_location"
        assert np.all(
            true_values.shape == self.model.theta_location.shape
        ), "true_values must have same dimensions as self.theta_location"

        return self._plot_coef_vs_ref(
            true_values=true_values,
            estim_values=self.model.theta_location,
            size=size,
            log=log,
            save=save,
            show=show,
            ncols=ncols,
            row_gap=row_gap,
            col_gap=col_gap,
            title="location_model",
            return_axs=return_axs,
        )

    def plot_coef_scale_vs_ref(
        self,
        true_values: np.ndarray,
        size=1,
        log=False,
        save=None,
        show=True,
        ncols=5,
        row_gap=0.3,
        col_gap=0.25,
        return_axs=False,
    ):
        """
        Plot estimated coefficients against reference (true) coefficients for scale model.

        :param true_values:
        :param size: Point size.
        :param save: Path+file name stem to save plots to.
            File will be save+"_genes.png". Does not save if save is None.
        :param show: Whether to display plot.
        :param ncols: Number of columns in plot grid if multiple genes are plotted.
        :param row_gap: Vertical gap between panel rows relative to panel height.
        :param col_gap: Horizontal gap between panel columns relative to panel width.
        :param return_axs: Whether to return axis objects.
        :return: Matplotlib axis objects.
        """
        assert len(true_values.shape) == len(
            self.model.theta_scale.shape
        ), "true_values must have same dimensions as self.theta_scale"
        assert np.all(
            true_values.shape == self.model.theta_scale.shape
        ), "true_values must have same dimensions as self.theta_scale"

        return self._plot_coef_vs_ref(
            true_values=true_values,
            estim_values=self.model.theta_scale,
            size=size,
            log=log,
            save=save,
            show=show,
            ncols=ncols,
            row_gap=row_gap,
            col_gap=col_gap,
            title="dispersion_model",
            return_axs=return_axs,
        )

    def plot_deviation_location(self, true_values: np.ndarray, save=None, show=True, return_axs=False):
        """
        Plot deviation of estimated coefficients from reference (true) coefficients
        as violin plot for location model.

        :param true_values:
        :param save: Path+file name stem to save plots to.
            File will be save+"_genes.png". Does not save if save is None.
        :param show: Whether to display plot.
        :param return_axs: Whether to return axis objects.
        :return: Matplotlib axis objects.
        """
        assert len(true_values.shape) == len(
            self.model.theta_location.shape
        ), "true_values must have same dimensions as self.theta_location"
        assert np.all(
            true_values.shape == self.model.theta_location.shape
        ), "true_values must have same dimensions as self.theta_location"

        return self._plot_deviation(
            true_values=true_values,
            estim_values=self.model.theta_location,
            save=save,
            show=show,
            title="location_model",
            return_axs=return_axs,
        )

    def plot_deviation_scale(self, true_values: np.ndarray, save=None, show=True, return_axs=False):
        """
        Plot deviation of estimated coefficients from reference (true) coefficients
        as violin plot for scale model.

        :param true_values:
        :param save: Path+file name stem to save plots to.
            File will be save+"_genes.png". Does not save if save is None.
        :param show: Whether to display plot.
        :param return_axs: Whether to return axis objects.
        :return: Matplotlib axis objects.
        """
        assert len(true_values.shape) == len(
            self.model.theta_scale.shape
        ), "true_values must have same dimensions as self.theta_scale"
        assert np.all(
            true_values.shape == self.model.theta_scale.shape
        ), "true_values must have same dimensions as self.theta_scale"

        return self._plot_deviation(
            true_values=true_values,
            estim_values=self.model.theta_scale,
            save=save,
            show=show,
            title="scale_model",
            return_axs=return_axs,
        )

    @property
    def loss(self) -> np.ndarray:
        """Current loss"""
        return self._loss

    @property
    def log_likelihood(self) -> np.ndarray:
        """Current log likelihood"""
        return self._log_likelihood

    @property
    def jacobian(self) -> np.ndarray:
        """Current Jacobian of the log likelihood"""
        return self._jacobian

    @property
    def hessian(self) -> np.ndarray:
        """Current Hessian of the log likelihood"""
        return self._hessian

    @property
    def fisher_inv(self) -> np.ndarray:
        """Current Fisher Inverse Matrix"""
        return self._fisher_inv

    @property
    def x(self) -> Union[np.ndarray, dask.array.core.Array]:
        """Data Matrix"""
        return self.model.x

    @property
    def theta_location(self):
        """Fit location parameter"""
        if isinstance(self.model.theta_location, dask.array.core.Array):
            return self.model.theta_location.compute()
        else:
            return self.model.theta_location

    @property
    def theta_scale(self) -> np.ndarray:
        """Fit scale parameter"""
        if isinstance(self.model.theta_scale, dask.array.core.Array):
            return self.model.theta_scale.compute()
        else:
            return self.model.theta_scale

    @abc.abstractmethod
    def initialize(self, **kwargs):
        """
        Initializes this estimator
        """
        pass

    def train_sequence(self, training_strategy, **kwargs):
        if isinstance(training_strategy, Enum):
            training_strategy = training_strategy.value
        elif isinstance(training_strategy, str):
            training_strategy = self.TrainingStrategies[training_strategy].value

        if training_strategy is None:
            training_strategy = self.TrainingStrategies.DEFAULT.value

        logger.debug("training strategy:\n%s", pprint.pformat(training_strategy))
        for idx, d in enumerate(training_strategy):
            logger.debug("Beginning with training sequence #%d", idx + 1)
            # Override duplicate arguments with user choice:
            if np.any([x in list(d.keys()) for x in list(kwargs.keys())]):
                d = dict([(x, y) for x, y in d.items() if x not in list(kwargs.keys())])
                for x in [xx for xx in list(d.keys()) if xx in list(kwargs.keys())]:
                    sys.stdout.write(
                        "overrding %s from training strategy with value %s with new value %s\n"
                        % (x, str(d[x]), str(kwargs[x]))
                    )
            self.train(**d, **kwargs)
            logger.debug("Training sequence #%d complete", idx + 1)

    @abc.abstractmethod
    def train(self, **kwargs):
        """
        Starts the training routine
        """
        pass

    @abc.abstractmethod
    def finalize(self, **kwargs):
        """
        Evaluate all tensors that need to be exported from session and save these as class attributes
        and close session.
        """
        pass

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
        return_axs=False,
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
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib import gridspec, rcParams

        if isinstance(true_values, dask.array.core.Array):
            true_values = true_values.compute()
        if isinstance(estim_values, dask.array.core.Array):
            estim_values = estim_values.compute()

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
            y = estim_values[i, :]
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
        else:
            return

    def _plot_deviation(
        self, true_values: np.ndarray, estim_values: np.ndarray, save=None, show=True, title=None, return_axs=False
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
        import matplotlib.pyplot as plt
        import seaborn as sns

        if isinstance(true_values, dask.array.core.Array):
            true_values = true_values.compute()
        if isinstance(estim_values, dask.array.core.Array):
            estim_values = estim_values.compute()

        plt.ioff()

        n_par = true_values.shape[0]
        summary_fit = pd.concat(
            [
                pd.DataFrame(
                    {
                        "deviation": estim_values[i, :] - true_values[i, :],
                        "coefficient": pd.Series(
                            ["coef_" + str(i) for x in range(estim_values.shape[1])], dtype="category"
                        ),
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
        else:
            return
