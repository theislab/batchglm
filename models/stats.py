import numpy as np


def normalize(measure: np.ndarray, data: np.ndarray) -> np.ndarray:
    """
    Normalize measure (e.g. `RMSD` or `MAE`) with the range of `data`

    :param measure: the measure which should be normalized
    :param data: ndarray representing the data by which the measure should be normalized
    :return: \frac{RMSD}{max(data) - min(data)}
    """
    norm = measure / (np.max(data) - np.min(data))
    return norm


def rmsd(estim: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """
    Calculate the root of the mean squared deviation between the estimated and the observed data

    :param estim: ndarray representing the estimated data
    :param obs: ndarray representing the observed data
    :return: \sqrt{mean{(estim - obs)^2}}
    """
    rmsd = np.sqrt(np.mean(np.square(estim - obs)))
    return rmsd


def mae(estim: np.ndarray, true_b: np.ndarray) -> np.ndarray:
    """
    Calculate the mean absolute error between the estimated weights `b` and the true `b`

    :param estim: ndarray representing the estimated data
    :param obs: ndarray representing the observed data
    :return: mean{(estim - obs)}
    """
    mae = np.mean(np.abs(estim - true_b))
    return mae


def normalized_rmsd(estim: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """
    Calculate the normalized RMSD between estimated and observed data

    :param estim: ndarray representing the estimated data
    :param obs: ndarray representing the observed data
    :return: \frac{RMSD}{max(obs) - min(obs)}
    """
    retval = normalize(rmsd(estim, obs), obs)
    return retval


def normalized_mae(estim: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """
    Calculate the normalized MAE between estimated and observed data

    :param estim: ndarray representing the estimated data
    :param obs: ndarray representing the observed data
    :return: \frac{MAE}{max(obs) - min(obs)}
    """
    retval = normalize(mae(estim, obs), obs)
    return retval
