import numpy as np


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
    return np.mean(np.abs(estim - obs) / obs, axis=axis)
