from typing import Union, Tuple

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions.negative_binomial import NegativeBinomial as TFNegativeBinomial

from batchglm.train.tf.ops import reduce_weighted_mean


class NegativeBinomial(TFNegativeBinomial):
    __mean: tf.Tensor
    __variance: tf.Tensor

    def __init__(self, total_count=None, r=None, variance=None, p=None, probs=None, log_probs=None, mean=None,
                 name="NegativeBinomial"):
        with tf.name_scope(name):
            if total_count is not None:  # total_count is alias for r
                r = total_count
            if log_probs is not None:  # calculate p from log_probs
                p = tf.exp(log_probs)
            if probs is not None:  # probs is alias for p
                p = probs

            if r is None:
                if variance is None:
                    raise ValueError("Must pass shape 'r' / 'total_count' or variance")

                if p is None:
                    if mean is None:
                        raise ValueError("Must pass 'probs' / 'p', 'log_probs' or 'mean'")

                    with tf.name_scope("reparametrize"):
                        p = 1. - (mean / variance)
                        p = tf.identity(p, "p")

                        r = mean * mean / (variance - mean)
                        r = tf.identity(r, "r")
            elif p is None:
                if mean is None:
                    raise ValueError("Must pass 'probs' / 'p', 'log_probs' or 'mean'")

                with tf.name_scope("reparametrize"):
                    p = mean / (r + mean)
                    p = tf.identity(p, "p")

            super().__init__(r, probs=p)

            self.__mean = mean
            self.__variance = variance

        self.__mean = mean
        self.__variance = variance

    def _mean(self) -> tf.Tensor:
        if self.__mean is None:
            self.__mean = super()._mean()

        return self.__mean

    def _variance(self) -> tf.Tensor:
        if self.__variance is None:
            self.__variance = super()._variance()

        return self.__variance

    @property
    def r(self) -> tf.Tensor:
        return self.total_count

    @property
    def p(self) -> tf.Tensor:
        return self.probs


def fit_partitioned(X: tf.Tensor, partitions: tf.Tensor,
                    axis=-2,
                    weights=None,
                    optimizable=False,
                    validate_shape=True,
                    dtype=tf.float32,
                    name="fit_nb_partitioned") -> NegativeBinomial:
    """
    Fits negative binomial distributions NB(r, p) to given sample data partitioned by 'design'.

    Usage example:
    `distribution = fit(X);`

    :param partitions: 1d vector of size 'shape(X)[axis] assigning each sample to one group/partition.
    :param X: matrix containing observations for each distribution on axis 'axis'\n
        E.g. `(N, M)` matrix with `M` distributions containing `N` observed values each
    :param axis: the axis containing the data of one distribution
    :param weights: if specified, the fit will be weighted
    :param optimizable: if true, the returned parameters will be optimizable
    :param validate_shape: should newly created variables perform shape inference?
    :param dtype: the data type to use; should be a floating point type
    :param name: name of the operation
    :return: negative binomial distribution
    """
    with tf.name_scope(name):
        uniques, partition_index = tf.unique(partitions, name="unique")

        def fit_part(i):
            smpl = tf.squeeze(tf.gather(X, tf.where(tf.equal(partition_index, i)), axis=axis), axis=axis)
            w = None
            if weights is not None:
                w = tf.squeeze(tf.gather(weights, tf.where(tf.equal(partition_index, i)), axis=axis), axis=axis)
            dist = fit(smpl, axis=axis, weights=w, optimizable=optimizable,
                       validate_shape=validate_shape, dtype=dtype)

            retval_tuple = (
                dist.r,
                # dist.p,
                dist.mean()
            )
            return retval_tuple

        idx = tf.range(tf.size(uniques))
        # r, p, mu = tf.map_fn(fit_part, idx, dtype=(X.dtype, X.dtype, X.dtype))
        r, mu = tf.map_fn(fit_part, idx, dtype=(dtype, dtype))

        # shape(r) == shape(mu) == [ size(uniques), ..., 1, ... ]

        stacked_r = tf.gather(r, partition_index, name="r")
        # stacked_p = tf.gather(p, partition_index, name="p")
        stacked_mu = tf.gather(mu, partition_index, name="mu")

        # shape(r) == shape(mu) == [ shape(X)[axis], ..., 1, ... ]

        with tf.name_scope("swap_dims"):
            perm = tf.range(tf.rank(r))[1:]
            perm = tf.concat(
                [
                    tf.expand_dims(perm[axis], 0),
                    tf.where(tf.equal(perm, perm[axis]), tf.zeros_like(perm), perm)
                ],
                axis=0
            )

        stacked_r = tf.squeeze(tf.transpose(stacked_r, perm=perm), axis=0)
        # stacked_p = tf.squeeze(tf.transpose(stacked_p, perm=perm), axis=0)
        stacked_mu = tf.squeeze(tf.transpose(stacked_mu, perm=perm), axis=0)

        # shape(r) == shape(mu) == shape(X)

        return NegativeBinomial(r=stacked_r, mean=stacked_mu)


def fit(X: Union[tf.Tensor, tf.SparseTensor], axis=0, weights=None, optimizable=False,
        validate_shape=True,
        dtype=tf.float32,
        name="nb-dist") -> Union[NegativeBinomial, Tuple[NegativeBinomial, tf.Variable, tf.Variable]]:
    """
    Fits negative binomial distributions NB(r, p) to given sample data along axis 'axis'.

    :param X: matrix containing observations for each distribution on axis 'axis'\n
        E.g. `(N, M)` matrix with `M` distributions containing `N` observed values each
    :param axis: the axis containing the data of one distribution
    :param weights: if specified, the closed-form fit will be weighted
    :param optimizable: if true, the returned distribution's parameters will be optimizable
    :param name: A name for the returned distribution (optional).
    :param validate_shape: should newly created variables perform shape inference?
    :param dtype: the data type to use; should be a floating point type
    :return: negative binomial distribution
    """
    with tf.name_scope("fit"):
        distribution = fit_mme(X, axis=axis, weights=weights, dtype=dtype)

        if optimizable:
            # alpha = tf.Variable(name="alpha", initial_value=tf.reciprocal(distribution.r),
            #                     dtype=dtype,
            #                     validate_shape=validate_shape)
            # alpha = tf.clip_by_value(alpha, tf.reciprocal(alpha.dtype.max), alpha.dtype.max, name="clipped_alpha")
            #
            # with tf.name_scope("variance"):
            #     variance = distribution.mean() + alpha * tf.square(distribution.mean())
            #
            # distribution = NegativeBinomial(variance=variance, mean=distribution.mean(), name=name)
            with tf.name_scope("r"):
                r_init = tf.clip_by_value(
                    tf.log(distribution.r),
                    clip_value_min=np.log(1),
                    clip_value_max=np.log(distribution.r.dtype.max) / 8
                )
                r_var = tf.Variable(name="log_r", initial_value=r_init,
                                    dtype=distribution.r.dtype,
                                    validate_shape=validate_shape)
                r = tf.clip_by_value(
                    r_var,
                    np.log(np.nextafter(0, 1, dtype=r_var.dtype.as_numpy_dtype)),
                    np.log(r_var.dtype.max)
                )
                r = tf.exp(r)
            with tf.name_scope("mu"):
                mu_init = tf.clip_by_value(
                    tf.log(distribution.mean()),
                    clip_value_min=np.log(1),
                    clip_value_max=np.log(distribution.mean().dtype.max) / 8
                )
                mu_var = tf.Variable(name="log_mu", initial_value=mu_init,
                                     dtype=distribution.mean().dtype,
                                     validate_shape=validate_shape)
                mu = tf.clip_by_value(
                    mu_var,
                    np.log(np.nextafter(0, 1, dtype=mu_var.dtype.as_numpy_dtype)),
                    np.log(mu_var.dtype.max)
                )
                mu = tf.exp(mu)

            return NegativeBinomial(r=r, mean=mu, name=name), mu_var, r_var
        else:
            return distribution


def fit_mme(X: Union[tf.Tensor, tf.SparseTensor], axis=0, weights=None, replace_values=None, dtype=tf.float32,
            name="MME") -> NegativeBinomial:
    """
        Calculates the Maximum-of-Momentum Estimator of `NB(r, p)` for given sample data along axis 'axis.

        :param X: matrix containing observations for each distribution on axis 'axis\n
            E.g. `(N, M)` matrix with `M` distributions containing `N` observed values each
        :param axis: the axis containing the data of one distribution
        :param weights: if specified, the fit will be weighted
        :param replace_values: Matrix of size `shape(X)[1:]`
        :param dtype: data type of replacement values, if `replace_values` is not set;
            If None, the replacement values will be of the same data type like `X`
        :param name: A name for the operation (optional).
        :return: estimated values of `r` and `p`
        """

    with tf.name_scope(name):
        mean = reduce_weighted_mean(X, weight=weights, axis=axis, keepdims=True, name="mean")
        variance = reduce_weighted_mean(tf.square(X - mean),
                                        weight=weights,
                                        axis=axis,
                                        keepdims=True,
                                        name="variance")
        if replace_values is None:
            replace_values = tf.fill(tf.shape(variance), tf.constant(variance.dtype.max, dtype=dtype),
                                     name="inf_constant")

        r = tf.where(tf.less(mean, variance),
                     tf.square(mean) / (variance - mean),
                     replace_values)
        r = tf.identity(r, "r")

        return NegativeBinomial(r=r, mean=mean)
