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

    zerofit = ((model.x < 1e-4) & (np.nan_to_num(model.location) < 1e-4)).compute()  # shape (obs, features)
    df_residual = resid_df(zerofit, model.design_loc)

    # Empirical Bayes squeezing of the quasi-likelihood variance factors
    s2 = nb_deviance(model) / df_residual
    s2[df_residual == 0] = 0.0  # s2[df.residual==0] <- 0
    s2 = np.maximum(s2, 0)  # s2 <- pmax(s2,0)

    df_prior, _, _ = squeeze_var(s2, df=df_residual, covariate=avg_log_cpm, robust=robust, winsor_tail_p=winsor_tail_p)
    return df_prior
