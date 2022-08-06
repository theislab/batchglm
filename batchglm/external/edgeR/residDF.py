import numpy as np


def combo_groups(truths: np.ndarray):
    """
    Function that returns a list of index lists, where each list refers to
    the rows with the same combination of TRUE/FALSE values in 'truths'.
    """
    uniq_cols, rev_index = np.unique(truths, axis=1, return_inverse=True)
    return [np.where(rev_index == i)[0] for i in range(uniq_cols.shape[1])]


def resid_df(zero: np.ndarray, design: np.ndarray):
    """
    Computes residual degrees of freedom.
    :param zero: boolean np.ndarray of shape (n_obs x n_features). It yields True if both
        the data and the fitted value of a GLM where close to zero within a small margin.
    :param design: the design matrix used in the GLM.
    """

    n_obs = zero.shape[0]
    n_param = design.shape[1]
    n_zero = np.sum(zero, axis=0)  # the number of zeros per feature; shape = (n_features, )

    degrees_of_freedom = np.full(zero.shape[1], n_obs - n_param)  # shape = (n_features, )
    degrees_of_freedom[n_zero == n_obs] = 0  # 0 if only zeros for specific feature

    some_zero_idx = (n_zero > 0) & (n_zero < n_obs)  # shape = (n_features, )
    if np.any(some_zero_idx):
        some_zero = zero[:, some_zero_idx]  # shape = (n_obs, len(np.where(some_zero_idx)))
        groupings = combo_groups(some_zero)  # list of idx in some_zero with identical cols

        degrees_of_freedom_some_zero = n_obs - n_zero[some_zero_idx]
        for group in groupings:
            some_zero_group = some_zero[:, group[0]]  # shape = (n_obs, )
            degrees_of_freedom_some_zero[group] -= np.linalg.matrix_rank(design[~some_zero_group].compute())
        degrees_of_freedom_some_zero = np.max(degrees_of_freedom_some_zero, 0)
        degrees_of_freedom[some_zero_idx] = degrees_of_freedom_some_zero

    return degrees_of_freedom
