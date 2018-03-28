import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.distributions import NegativeBinomial as TFNegativeBinomial

__all__ = ['fit_mme', 'fit', 'fit_partitioned', 'NegativeBinomial']


class NegativeBinomial(TFNegativeBinomial):
    mu: tf.Tensor
    p: tf.Tensor
    r: tf.Tensor

    def __init__(self, r, p=None, mu=None, name="NegativeBinomial"):
        with tf.name_scope(name):
            if p is not None:
                if mu is not None:
                    raise ValueError("Must pass either probs or means, but not both")

                with tf.name_scope("reparametrize"):
                    mu = p * r / (1 - p)
                    mu = tf.identity(mu, "mu")

                super().__init__(r, probs=p, name=name)
            elif mu is not None:
                if p is not None:
                    raise ValueError("Must pass either probs or means, but not both")
                # p is directly dependent from mu and r
                self.mu = mu
                with tf.name_scope("reparametrize"):
                    p = mu / (r + mu)
                    p = tf.identity(p, "p")
                super().__init__(r, probs=p)
            else:
                raise ValueError("Must pass probs or means")
        self.mu = mu
        self.p = p
        self.r = r


def fit_partitioned(sample_data, design, optimizable=False, name="fit_nb_partitioned") -> NegativeBinomial:
    """
    Fits negative binomial distributions NB(r, p) to given sample data partitioned by 'design'.

    Usage example:
    `distribution = fit(sample_data);`

    :param design: 1d vector of size 'shape(sample_data)[0] assigning each sample to one group/partition.
    :param sample_data: matrix containing samples for each distribution on axis 0\n
        E.g. `(N, M)` matrix with `M` distributions containing `N` observed values each
    :param optimizable: if true, the returned parameters will be optimizable
    :param name: name of the operation
    :return: negative binomial distribution
    """
    with tf.name_scope(name):
        design_df = pd.DataFrame(design)
        design_df["row_nr"] = design_df.index

        # create DataFrame containing all unique rows and assign a unique index to them
        unique_rows = np.unique(design, axis=0)
        unique_rows = pd.DataFrame(unique_rows)
        unique_rows["idx"] = unique_rows.index
        # merge the unique row indexes to the design matrix
        indexed_design = design_df.merge(unique_rows)
        indexed_design = indexed_design.sort_values(by=['row_nr'])

        partition_index = list(indexed_design["idx"])

        dyn_parts = tf.dynamic_partition(sample_data, partition_index, unique_rows.shape[0], name="dynamic_parts")
        params_r = list()
        params_p = list()
        params_mu = list()
        for idx, part in enumerate(dyn_parts):
            dist = fit(part, optimizable=optimizable, name="negbin_%i" % idx)
            params_r.append(dist.total_count)
            params_p.append(dist.probs)
            params_mu.append(dist.mu)

        stacked_r = tf.stack(params_r, name="stack_r")
        stacked_p = tf.stack(params_p, name="stack_p")

        stacked_r = tf.gather(stacked_r, partition_index, name="r")
        stacked_p = tf.gather(stacked_p, partition_index, name="p")

        return NegativeBinomial(r=stacked_r, p=stacked_p)


def fit(sample_data, optimizable=False, name="nb-dist") -> NegativeBinomial:
    """
    Fits negative binomial distributions NB(r, p) to given sample data along axis 0.

    :param sample_data: matrix containing samples for each distribution on axis 0\n
        E.g. `(N, M)` matrix with `M` distributions containing `N` observed values each
    :param optimizable: if true, the returned distribution's parameters will be optimizable
    :param name: A name for the returned distribution (optional).
    :return: negative binomial distribution
    """
    with tf.name_scope("fit"):
        (r, p) = fit_mme(sample_data)

        if optimizable:
            r = tf.Variable(name="r", initial_value=r, dtype=tf.float32, validate_shape=False)
            # r_var = tf.Variable(tf.zeros(tf.shape(r)), dtype=tf.float32, validate_shape=False, name="r_var")
            #
            # r_assign_op = tf.assign(r_var, r)

        # keep mu constant
        mu = tf.reduce_mean(sample_data, axis=0, name="mu")

        distribution = NegativeBinomial(r=r, mu=mu, name=name)

    return distribution


def fit_mme(sample_data, replace_values=None, dtype=None, name="MME") -> (tf.Tensor, tf.Tensor):
    """
        Calculates the Maximum-of-Momentum Estimator of `NB(r, p)` for given sample data along axis 0.

        :param sample_data: matrix containing samples for each distribution on axis 0\n
            E.g. `(N, M)` matrix with `M` distributions containing `N` observed values each
        :param replace_values: Matrix of size `shape(sample_data)[1:]`
        :param dtype: data type of replacement values, if `replace_values` is not set;
            If None, the replacement values will be of the same data type like `sample_data`
        :param name: A name for the operation (optional).
        :return: estimated values of `r` and `p`
        """
    if dtype is None:
        dtype = sample_data.dtype

    with tf.name_scope(name):
        mean = tf.reduce_mean(sample_data, axis=0, name="mean")
        variance = tf.reduce_mean(tf.square(sample_data - mean),
                                  axis=0,
                                  name="variance")
        if replace_values is None:
            replace_values = tf.fill(tf.shape(variance), tf.constant(math.inf, dtype=dtype), name="inf_constant")

        r_by_mean = tf.where(tf.less(mean, variance),
                             mean / (variance - mean),
                             replace_values)
        r = r_by_mean * mean
        r = tf.identity(r, "r")

        p = 1 / (r_by_mean + 1)
        p = tf.identity(p, "p")

        return r, p
