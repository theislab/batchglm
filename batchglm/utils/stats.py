import numpy as np
import scipy.stats


def normalize(measure: np.ndarray, data: np.ndarray) -> np.ndarray:
    """
    Normalize measure (e.g. `RMSD` or `MAE`) with the range of `data`

    :param measure: the measure which should be normalized
    :param data: Tensor representing the data by which the measure should be normalized
    :return: \frac{RMSD}{max(data) - min(data)}
    """
    return measure / (np.max(data) - np.min(data))


def rmsd(estim: np.ndarray, obs: np.ndarray, axis=None) -> np.ndarray:
    """
    Calculate the root of the mean squared deviation between the estimated and the observed data

    :param estim: Tensor representing the estimated data
    :param obs: Tensor representing the observed data
    :param axis: axis to reduce
    :return: \sqrt{mean{(estim - obs)^2}}
    """
    return np.sqrt(np.mean(np.square(estim - obs), axis=axis))


def mae(estim: np.ndarray, obs: np.ndarray, axis=None) -> np.ndarray:
    """
    Calculate the mean absolute error between the estimated weights `b` and the true `b`

    :param estim: Tensor representing the estimated data
    :param obs: Tensor representing the observed data
    :param axis: axis to reduce
    :return: mean{|estim - obs|}
    """
    return np.mean(np.abs(estim - obs), axis=axis)


def normalized_rmsd(estim: np.ndarray, obs: np.ndarray, axis=None) -> np.ndarray:
    """
    Calculate the normalized RMSD between estimated and observed data

    :param estim: Tensor representing the estimated data
    :param obs: Tensor representing the observed data
    :param axis: axis to reduce
    :return: \frac{RMSD}{max(obs) - min(obs)}
    """
    return normalize(rmsd(estim, obs, axis=axis), obs)


def normalized_mae(estim: np.ndarray, obs: np.ndarray, axis=None) -> np.ndarray:
    """
    Calculate the normalized MAE between estimated and observed data

    :param estim: Tensor representing the estimated data
    :param obs: Tensor representing the observed data
    :param axis: axis to reduce
    :return: \frac{MAE}{max(obs) - min(obs)}
    """
    return normalize(mae(estim, obs, axis=axis), obs)


def mapd(estim: np.ndarray, obs: np.ndarray, axis=None) -> np.ndarray:
    """
    Calculate the mean absolute percentage deviation between the estimated and the observed data

    :param estim: ndarray representing the estimated data
    :param obs: ndarray representing the observed data
    :param axis: axis to reduce
    :return: mean{|estim - obs| / obs}
    """
    return np.mean(abs_percentage_deviation(estim, obs), axis=axis) * 100


def abs_percentage_deviation(estim: np.ndarray, obs: np.ndarray) -> np.ndarray:
    r"""
    Calculate the absolute percentage deviation between the estimated and the observed data

    :param estim: ndarray representing the estimated data
    :param obs: ndarray representing the observed data
    :return: $\mean{|estim - obs| / |obs|} * 100$
    """
    return abs_proportional_deviation(estim, obs)


def abs_proportional_deviation(estim: np.ndarray, obs: np.ndarray) -> np.ndarray:
    r"""
    Calculate the absolute proportional deviation between the estimated and the observed data

    :param estim: ndarray representing the estimated data
    :param obs: ndarray representing the observed data
    :return: mean{|estim - obs| / |obs|}
    """
    return np.abs(estim - obs) / np.abs(obs)


def welch(mu1, mu2, var1, var2, n1, n2):
    s_delta = np.sqrt((var1 / n1) + (var2 / n2))
    t = (mu1 - mu2) / s_delta

    df = (
            np.square((var1 / n1) + (var2 / n2)) /
            (
                    (np.square(var1 / n1) / (n1 - 1)) +
                    (np.square(var2 / n2) / (n2 - 1))
            )
    )

    return t, df


def welch_t_test(x1, x2):
    """
    Calculates a Welch-Test with independent mean and variance for two samples.
    Tests the null hypothesis (H0) that the two population means are equal.

    :param x1: first sample
    :param x2: second sample
    :return: p-value
    """
    mu1 = np.mean(x1)
    var1 = np.var(x1)
    mu2 = np.mean(x2)
    var2 = np.var(x2)
    n1 = np.size(x1)
    n2 = np.size(x2)

    t, df = welch(mu1, mu2, var1, var2, n1, n2)
    pval = 1 - scipy.stats.t(df).cdf(t)

    return pval
