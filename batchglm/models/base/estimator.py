import abc
from abc import ABC
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import dask
import logging
import numpy as np
import pandas as pd
import pprint
import sys

from .input import InputDataBase
from .model import _ModelBase
from .external import maybe_compute, types as T

logger = logging.getLogger(__name__)


# TODO: why abc.Meta instead of abc.ABC
class _EstimatorBase(metaclass=abc.ABCMeta):
    r"""
    Estimator base class
    """
    # TODO: why are these here?
    model: _ModelBase
    _loss: np.ndarray  # is this really an array?
    _jacobian: np.ndarray

    # TODO: better enums (use the pretty-printing like in CellRank)
    # TODO: don't use nested classes
    class TrainingStrategy(Enum):
        AUTO = None

    def __init__(
            self,
            model: _ModelBase,
            input_data: InputDataBase
    ):
        self.model = model
        self.input_data = input_data
        self._loss = None
        self._log_likelihood = None
        self._jacobian = None
        self._hessian = None
        self._fisher_inv = None
        self._error_codes = None
        self._niter = None

    # TODO: type
    @property
    def error_codes(self):
        return self._error_codes

    @property
    def niter(self) -> int:
        return self._niter

    @property
    def loss(self) -> np.ndarray:
        return self._loss

    @property
    def log_likelihood(self):
        return self._log_likelihood

    @property
    def jacobian(self) -> np.ndarray:
        return self._jacobian

    @property
    def hessian(self) -> np.ndarray:
        return self._hessian

    # TODO: type
    @property
    def fisher_inv(self):
        return self._fisher_inv

    @property
    def x(self) -> np.ndarray:
        return self.input_data.x

    @property
    def w(self) -> np.ndarray:
        return self.input_data.w

    @property
    def a_var(self) -> np.ndarray:
        return maybe_compute(self.model.a_var)

    @property
    def b_var(self) -> np.ndarray:
        return maybe_compute(self.model.b_var)

    @abc.abstractmethod
    def initialize(self, **kwargs):
        """
        Initializes this estimator
        """
        pass

    # TODO: type training strategy
    # TODO: docs
    def train_sequence(
            self,
            training_strategy,
            **kwargs
    ):
        # TODO: better enums (use the pretty-printing like in CellRank)
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
                    # TODO: don't use sys
                    sys.stdout.write(
                        "overrding %s from training strategy with value %s with new value %s\n" %
                        (x, str(d[x]), str(kwargs[x]))
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

    # TODO: make static, not use model?
    def _plot_coef_vs_ref(
            self,
            true_values: T.ArrayLike,
            estim_values: T.ArrayLike,
            size: float = 1,
            log: bool = False,
            save: Optional[Union[str, Path]] = None,
            show: bool = True,
            ncols: int = 5,
            row_gap: float = 0.3,
            col_gap: float = 0.25,
            title: Optional[str] = None,
            return_axs: bool = False
    ) -> Optional['matplotlib.axes.Axes']:
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

        true_values = maybe_compute(true_values)
        estim_values = maybe_compute(estim_values)

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

    # TODO: make static, not use model?
    def _plot_deviation(
            self,
            true_values: T.ArrayLike,
            estim_values: T.ArrayLike,
            save: Optional[Union[str, Path]] = None,
            show: bool = True,
            title: Optional[str] = None,
            return_axs: bool = False
    ) -> Optional['matplotlib.axes.Axes']:
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

        if isinstance(true_values, dask.array.core.Array):
            true_values = true_values.compute()
        if isinstance(estim_values, dask.array.core.Array):
            estim_values = estim_values.compute()

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


class EstimatorBaseTyping(_EstimatorBase, ABC):
    r"""
    Estimator base class used for typing in other packages.
    """

