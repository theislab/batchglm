import abc
import numpy as np

try:
    import anndata
except ImportError:
    anndata = None

from .external import _EstimatorBase
from .input import InputDataGLM
from .model import _ModelGLM


class _EstimatorGLM(_EstimatorBase, metaclass=abc.ABCMeta):
    r"""
    Estimator base class for generalized linear models (GLMs).
    """
    model: _ModelGLM
    input_data: InputDataGLM

    def __init__(
            self,
            model: _ModelGLM,
            input_data: InputDataGLM
    ):
        _EstimatorBase.__init__(
            self=self,
            model=model,
            input_data=input_data
        )

    def plot_coef_a_vs_ref(
            self,
            true_values: np.ndarray,
            size=1,
            log=False,
            save=None,
            show=True,
            ncols=5,
            row_gap=0.3,
            col_gap=0.25,
            return_axs=False
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
        assert len(true_values.shape) == len(self.model.a_var.shape), \
            "true_values must have same dimensions as self.a_var"
        assert np.all(true_values.shape == self.model.a_var.shape), \
            "true_values must have same dimensions as self.a_var"

        return self._plot_coef_vs_ref(
            true_values=true_values,
            estim_values=self.model.a_var,
            size=size,
            log=log,
            save=save,
            show=show,
            ncols=ncols,
            row_gap=row_gap,
            col_gap=col_gap,
            title="location_model",
            return_axs=return_axs
        )

    def plot_coef_b_vs_ref(
            self,
            true_values: np.ndarray,
            size=1,
            log=False,
            save=None,
            show=True,
            ncols=5,
            row_gap=0.3,
            col_gap=0.25,
            return_axs=False
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
        assert len(true_values.shape) == len(self.model.b_var.shape), \
            "true_values must have same dimensions as self.b_var"
        assert np.all(true_values.shape == self.model.b_var.shape), \
            "true_values must have same dimensions as self.b_var"

        return self._plot_coef_vs_ref(
            true_values=true_values,
            estim_values=self.model.b_var,
            size=size,
            log=log,
            save=save,
            show=show,
            ncols=ncols,
            row_gap=row_gap,
            col_gap=col_gap,
            title="dispersion_model",
            return_axs=return_axs
        )

    def plot_deviation_a(
            self,
            true_values: np.ndarray,
            save=None,
            show=True,
            return_axs=False
    ):
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
        assert len(true_values.shape) == len(self.model.a_var.shape), \
            "true_values must have same dimensions as self.a_var"
        assert np.all(true_values.shape == self.model.a_var.shape), \
            "true_values must have same dimensions as self.a_var"

        return self._plot_deviation(
            true_values=true_values,
            estim_values=self.model.a_var,
            save=save,
            show=show,
            title="location_model",
            return_axs=return_axs
        )

    def plot_deviation_b(
            self,
            true_values: np.ndarray,
            save=None,
            show=True,
            return_axs=False
    ):
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
        assert len(true_values.shape) == len(self.model.b_var.shape), \
            "true_values must have same dimensions as self.b_var"
        assert np.all(true_values.shape == self.model.b_var.shape), \
            "true_values must have same dimensions as self.b_var"

        return self._plot_deviation(
            true_values=true_values,
            estim_values=self.model.b_var,
            save=save,
            show=show,
            title="scale_model",
            return_axs=return_axs
        )
