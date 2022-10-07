from typing import Optional, Tuple, Union

import dask.array
import numpy as np

from .c_utils import nb_deviance
from .estimator import NBEstimator
from .external import NBModel
from .limma import squeeze_var
from .residDF import resid_df


def calculate_prior_df(
    model: NBModel,
    robust: bool,
    dispersion: Union[np.ndarray, float],
    winsor_tail_p: Tuple[float, float],
    avg_log_cpm: Optional[np.ndarray] = None,
    tolerance: float = 1e-10,
):
    """
    Calculates prior degrees of freedom. This is a wrapper function around limma's squeezeVar.
    This is a python version of edgeR's priorDF function.
    """
    estimator = NBEstimator(model, dispersion=dispersion)
    estimator.train(maxit=250, tolerance=tolerance)

    fitted_model = estimator._model_container
    loc = fitted_model.location
    scale = fitted_model.scale
    x = fitted_model.x
    dloc = fitted_model.design_loc

    zerofit = (x < 1e-4) & (np.nan_to_num(loc) < 1e-4)
    if isinstance(zerofit, dask.array.core.Array):
        zerofit = zerofit.compute()  # shape (obs, features)
    if isinstance(dloc, dask.array.core.Array):
        dloc = dloc.compute()
    df_residual = resid_df(zerofit, dloc)

    # Empirical Bayes squeezing of the quasi-likelihood variance factors
    if isinstance(x, dask.array.core.Array):
        x = x.compute()
    if isinstance(loc, dask.array.core.Array):
        loc = loc.compute()
    if isinstance(scale, dask.array.core.Array):
        scale = scale.compute()

    with np.errstate(divide="ignore"):
        s2 = nb_deviance(x, loc, scale, True) / df_residual
    s2[df_residual == 0] = 0.0
    s2 = np.maximum(s2, 0)

    return squeeze_var(s2, df=df_residual, covariate=avg_log_cpm, robust=robust, winsor_tail_p=winsor_tail_p)
