import abc
import numpy as np

try:
    import anndata
except ImportError:
    anndata = None

from .model import MODEL_PARAMS
from .external import _Estimator_Base, _EstimatorStore_XArray_Base

ESTIMATOR_PARAMS = MODEL_PARAMS.copy()
ESTIMATOR_PARAMS.update({
    "loss": (),
    "log_likelihood": ("features",),
    "gradients": ("features",),
    "hessians": ("features", "delta_var0", "delta_var1"),
    "fisher_inv": ("features", "delta_var0", "delta_var1"),
})

class _Estimator_GLM(_Estimator_Base, metaclass=abc.ABCMeta):
    r"""
    Estimator base class for generalized linear models (GLMs).
    """

class _EstimatorStore_XArray_GLM(_EstimatorStore_XArray_Base):

    def __init__(self):
        super(_EstimatorStore_XArray_Base, self).__init__()

    @property
    def log_likelihood(self):
        return self.params["log_likelihood"]

    @property
    def gradients(self):
        return self.params["gradients"]

    @property
    def hessians(self):
        return self.params["hessians"]

    @property
    def fisher_inv(self):
        return self.params["fisher_inv"]

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
        Plot estimated coefficients against reference (true) coefficients.

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

        return self._plot_coef_vs_ref(
            true_values=true_values,
            estim_values=self.a,
            size=size,
            log=log,
            save=save,
            show=show,
            ncols=ncols,
            row_gap=row_gap,
            col_gap=col_gap,
            title="location_parameter",
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
        Plot estimated coefficients against reference (true) coefficients.

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

        return self._plot_coef_vs_ref(
            true_values=true_values,
            estim_values=self.b,
            size=size,
            log=log,
            save=save,
            show=show,
            ncols=ncols,
            row_gap=row_gap,
            col_gap=col_gap,
            title="dispersion_parameter",
            return_axs=return_axs
        )
