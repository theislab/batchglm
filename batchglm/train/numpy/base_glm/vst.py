from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def log_geometric_mean(a: np.ndarray, axis: Optional[int] = None, eps: Union[int, float] = 1.0):
    """
    Returns the log of the geometric mean defined as mean(log(a + eps)).
    :param a: np.ndarray containing the data
    :param axis: the axis over which the log geometric mean is calculated. If None, computes the log geometric mean
        over the entire data.
    :param eps: small value added to each value in order to avoid computing log(0). Default is 1, i.e. log(x) is
        equivalent to np.log1p(x).
    :return np.ndarray: An array with the same length as the axis not used for computation containing the log geometric
         means.
    """
    log_a = np.log(a + eps)
    return np.mean(log_a, axis=axis)


def geometric_mean(a: np.ndarray, axis: Optional[int] = None, eps: Union[int, float] = 1.0):
    r"""
    Return the geometric mean defined as (\prod_{i=1}^{n}(x_{i} + eps))^{1/n} - eps computed in log space as
    exp(1/n(sum_{i=1}^{n}(ln(x_{i} + eps)))) - eps for numerical stability.
    :param a: np.ndarray containing the data
    :param axis: the axis over which the geometric mean is calculated. If None, computes the geometric mean over the
        entire data.
    :param eps: small value added to each value in order to avoid computing log(0). Default is 1, i.e. log(x) is
        equivalent to np.log1p(x).
    :return np.ndarray: An array with the same length as the axis not used for computation containing the geometric
        means.
    """
    log_geo_mean = log_geometric_mean(a, axis, eps)
    return np.exp(log_geo_mean) - eps


def bw_kde(x: np.ndarray, method: str = "silverman"):
    """
    Performs gaussian kernel density estimation using a specified method and returns the estimated bandwidth.
    :param x: np.ndarray of values for which the KDE is performed.
    :param method: The method used for estimating an optimal bandwidth, see scipy gaussian_kde documentation for
        available methods.
    :return float: The estimated bandwidth
    """
    return gaussian_kde(x, bw_method=method).factor * 0.37
    # return FFTKDE(kernel=kernel, bw=method).fit(x).bw


def robust_scale(x: pd.Series, c: float = 1.4826, eps: Optional[Union[int, float]] = None):
    r"""
    Compute a scale param using the formula scale_{i} = (x_{i} - median(x)) / mad(x) where
    mad = c * median(abs(x_{i} - median(x))) + eps is the median absolute deviation.
    This function is derived from sctransform's implementation of robust scale:
        https://github.com/satijalab/sctransform/blob/7e9c9557222d1e34416e8854ed22da580e533e78/R/utils.R#L162-L163
    :param x: pd.Series containing the values used to compute the scale.
    :param c: Scaling constant used in the computation of mad. The default value is equivalent to the R implementation
        of mad: https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/mad
    :param eps: Small value added to the mad. If None, it defaults to `np.finfo(float).eps`. This should be equivalent
        to sctransform's `.Machine$double.eps`.

    :return pd.Series containing the computed scales.
    """
    if eps is None:
        eps = float(np.finfo(float).eps)

    deviation = x - x.median()
    mad = c * deviation.abs().median() + eps
    scale = deviation / mad
    return scale


def is_outlier(model_param: np.ndarray, means: np.ndarray, threshold: Union[int, float] = 10):
    """
    Compute outlier genes based on deviation of model_param from mean counts in individual bins.
    This function is derived from sctransform's implementation of is_outlier:
    https://github.com/satijalab/sctransform/blob/7e9c9557222d1e34416e8854ed22da580e533e78/R/utils.R#L120-L129
    :param model_param: np.ndarray of a specific model_param. This can be the intercept, any batch/condition or
        loc/scale param. This is the param based on which it is determined if a specific gene is an outlier.
    :param means: np.ndarray of genewise mean counts. The means are used to determine bins within outlier detection of
        the model_param is performed.
    :param threshold: The minimal score required for a model_param to be considered an outlier.

    :return np.ndarray of booleans indicating if a particular gene is an outlier (True) or not (False).
    """
    bin_width = (means.max() - means.min()) * bw_kde(means) / 2
    eps = np.finfo(float).eps * 10

    breaks1 = np.arange(means.min() - eps, means.max() + bin_width, bin_width)
    breaks2 = np.arange(means.min() - eps - bin_width / 2, means.max() + bin_width, bin_width)
    bins1 = pd.cut(means, bins=breaks1)
    bins2 = pd.cut(means, bins=breaks2)

    df_tmp = pd.DataFrame({"param": model_param, "bins1": bins1, "bins2": bins2})
    df_tmp["score1"] = df_tmp.groupby("bins1")["param"].transform(robust_scale)
    df_tmp["score2"] = df_tmp.groupby("bins2")["param"].transform(robust_scale)
    df_tmp["outlier"] = df_tmp[["score1", "score2"]].abs().min(axis=1) > threshold

    return df_tmp
