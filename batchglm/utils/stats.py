import numpy as np
import scipy.stats


def normalize(measure: np.ndarray, data: np.ndarray) -> np.ndarray:
    r"""
    Normalize measure (e.g. :math:`RMSD` or :math:`MAE`) with the range of :math:`data`

    :param measure: the measure which should be normalized
    :param data: Tensor representing the data by which the measure should be normalized
    :return: :math:`\frac{RMSD}{\text{max}(\text{data}) - \text{min}(\text{data})}`

    r"""
    return measure / (np.max(data) - np.min(data))


def rmsd(estim: np.ndarray, obs: np.ndarray, axis=None) -> np.ndarray:
    r"""
    Calculate the root of the mean squared deviation between the estimated and the observed data

    :param estim: Tensor representing the estimated data
    :param obs: Tensor representing the observed data
    :param axis: axis to reduce
    :return: :math:`\sqrt{\text{mean}{(\text{estim} - \text{obs})^2}}`

    r"""
    return np.sqrt(np.mean(np.square(estim - obs), axis=axis))


def mae(estim: np.ndarray, obs: np.ndarray, axis=None) -> np.ndarray:
    r"""
    Calculate the mean absolute error between the estimated weights :math:`b` and the true :math:`b`

    :param estim: Tensor representing the estimated data
    :param obs: Tensor representing the observed data
    :param axis: axis to reduce
    :return: :math:`\text{mean}{|\text{estim} - \text{obs}|}`

    r"""
    return np.mean(np.abs(estim - obs), axis=axis)


def normalized_rmsd(estim: np.ndarray, obs: np.ndarray, axis=None) -> np.ndarray:
    r"""
    Calculate the normalized RMSD between estimated and observed data

    :param estim: Tensor representing the estimated data
    :param obs: Tensor representing the observed data
    :param axis: axis to reduce
    :return: :math:`\frac{RMSD}{\text{max}(\text{obs}) - \text{min}(\text{obs})}`

    r"""
    return normalize(rmsd(estim, obs, axis=axis), obs)


def normalized_mae(estim: np.ndarray, obs: np.ndarray, axis=None) -> np.ndarray:
    r"""
    Calculate the normalized MAE between estimated and observed data

    :param estim: Tensor representing the estimated data
    :param obs: Tensor representing the observed data
    :param axis: axis to reduce
    :return: :math:`\frac{MAE}{\text{max}(\text{obs}) - \text{min}(\text{obs})}`

    r"""
    return normalize(mae(estim, obs, axis=axis), obs)


def mapd(estim: np.ndarray, obs: np.ndarray, axis=None) -> np.ndarray:
    r"""
    Calculate the mean absolute percentage deviation between the estimated and the observed data

    :param estim: ndarray representing the estimated data
    :param obs: ndarray representing the observed data
    :param axis: axis to reduce
    :return: :math:`\text{mean}(|\text{estim} - \text{obs}| / |\text{obs}|)`

    r"""
    return np.mean(abs_percentage_deviation(estim, obs), axis=axis) * 100


def abs_percentage_deviation(estim: np.ndarray, obs: np.ndarray) -> np.ndarray:
    r"""
    Calculate the absolute percentage deviation between the estimated and the observed data

    :param estim: ndarray representing the estimated data
    :param obs: ndarray representing the observed data
    :return: :math:`\text{mean}(|\text{estim} - \text{obs}| / | \text{obs}|) * 100`

    r"""
    return abs_proportional_deviation(estim, obs)


def abs_proportional_deviation(estim: np.ndarray, obs: np.ndarray) -> np.ndarray:
    r"""
    Calculate the absolute proportional deviation between the estimated and the observed data

    :param estim: ndarray representing the estimated data
    :param obs: ndarray representing the observed data
    :return: :math:`\text{mean}{|\text{estim} - \text{obs}| / |\text{obs}|}`

    r"""
    return np.abs(estim - obs) / np.abs(obs)


def welch(mu1, mu2, var1, var2, n1, n2):
    s_delta = np.sqrt((var1 / n1) + (var2 / n2))
    s_delta = np.asarray(s_delta)
    s_delta = np.nextafter(0, 1, out=s_delta, where=s_delta == 0, dtype=s_delta.dtype)

    t = (mu1 - mu2) / s_delta
    # t = np.asarray(t)
    # t = np.nextafter(0, 1, out=t, where=t == 0, dtype=t.dtype)

    denom = (np.square(var1 / n1) / (n1 - 1)) + (np.square(var2 / n2) / (n2 - 1))
    denom = np.asarray(denom)
    denom = np.nextafter(0, 1, out=denom, where=denom == 0, dtype=denom.dtype)

    df = np.square((var1 / n1) + (var2 / n2)) / denom

    return t, df


def welch_t_test(x1, x2):
    r"""
    Calculates a Welch-Test with independent mean and variance for two samples.
    Tests the null hypothesis (H0) that the two population means are equal.

    :param x1: first sample
    :param x2: second sample
    :return: p-value

    r"""
    mu1 = np.mean(x1)
    var1 = np.var(x1)
    mu2 = np.mean(x2)
    var2 = np.var(x2)
    n1 = np.size(x1)
    n2 = np.size(x2)

    t, df = welch(mu1, mu2, var1, var2, n1, n2)
    pval = 1 - scipy.stats.t(df).cdf(t)

    if np.isnan(pval):
        return 1.

    return pval
