import numpy as np

from .fitFDist import fit_f_dist, fit_f_dist_robustly


def squeeze_var(var: np.ndarray, df: np.ndarray, covariate: np.ndarray, robust: bool, winsor_tail_p: np.ndarray):
    n = len(var)
    # 	Degenerate special cases
    if n == 1:
        return var, var, 0

    # 	When df==0, guard against missing or infinite values in var
    if len(df) > 1:
        var[df == 0] = 0

    # 	Estimate hyperparameters
    if robust:
        fit = fit_f_dist_robustly(var=var, df1=df, covariate=covariate, winsor_tail_p=winsor_tail_p)
        df_prior = fit.df2_shrunk
    else:
        fit = fit_f_dist(var, df1=df, covariate=covariate)
        df_prior = fit.df2

    if np.any(np.isnan(df_prior)):
        raise ValueError("Could not estimate prior df due to NaN")

    # 	Posterior variances
    var_post = _squeeze_var(var=var, df=df, var_prior=fit.scale, df_prior=df_prior)

    return df_prior, fit.scale, var_post


def _squeeze_var(var: np.ndarray, df: np.ndarray, var_prior: np.ndarray, df_prior: np.ndarray):
    """
    Squeeze posterior variances given hyperparameters
    """

    n = len(var)
    isfin = np.isfinite(df_prior)
    if np.all(isfin):
        return (df * var + df_prior * var_prior) / (df + df_prior)

    # 	From here, at least some df.prior are infinite

    # 	For infinite df_prior, return var_prior
    if len(var_prior) == n:
        var_post = var_prior
    else:
        var_post = np.full(n, var_prior)

    # 	Maybe some df.prior are finite
    if np.any(isfin):
        if len(df) > 1:
            df = df[isfin]
        df_prior = df_prior[isfin]
        var_post[isfin] = (df * var[isfin] + df_prior * var_post[isfin]) / (df + df_prior)

    return var_post
