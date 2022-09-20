import dask.array
import numpy as np

from .c_utils import nb_deviance
from .estimator import NBEstimator
from .external import BaseModelContainer
from .limma import squeeze_var
from .residDF import resid_df


def calculate_prior_df(
    model: BaseModelContainer,
    avg_log_cpm: np.ndarray,
    robust: bool,
    winsor_tail_p: np.ndarray,
    dispersion: np.ndarray,
    tolerance: float = 1e-10,
):
    """
    Calculates prior degrees of freedom. This is a wrapper function around limma's squeezeVar.
    This is a python version of edgeR's priorDF function.
    """
    estimator = NBEstimator(model, dispersion=dispersion)
    estimator.train(maxit=250, tolerance=tolerance)

    zerofit = (model.x < 1e-4) & (np.nan_to_num(model.location) < 1e-4)
    if isinstance(zerofit, dask.array.core.Array):
        zerofit = zerofit.compute()  # shape (obs, features)
    dloc = model.design_loc
    if isinstance(model.design_loc, dask.array.core.Array):
        dloc = dloc.compute()
    df_residual = resid_df(zerofit, dloc)

    # Empirical Bayes squeezing of the quasi-likelihood variance factors
    x = model.x
    if isinstance(model.x, dask.array.core.Array):
        x = x.compute()
    loc = model.location
    if isinstance(model.location, dask.array.core.Array):
        loc = loc.compute()
    scale = model.scale
    if isinstance(model.scale, dask.array.core.Array):
        scale = scale.compute()

    with np.errstate(divide="ignore"):
        s2 = nb_deviance(x, loc, scale, True) / df_residual
    s2[df_residual == 0] = 0.0
    s2 = np.maximum(s2, 0)

    return squeeze_var(s2, df=df_residual, covariate=avg_log_cpm, robust=robust, winsor_tail_p=winsor_tail_p)
