import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import NegativeBinomial as TFNegativeBinomial

from impl.tf.util import reduce_weighted_mean


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


def fit_partitioned(sample_data: tf.Tensor, partitions: tf.Tensor,
                    axis=-2,
                    weights=None,
                    optimizable=False,
                    validate_shape=True,
                    dtype=tf.float32,
                    name="fit_nb_partitioned") -> NegativeBinomial:
    """
    Fits negative binomial distributions NB(r, p) to given sample data partitioned by 'design'.

    Usage example:
    `distribution = fit(sample_data);`

    :param partitions: 1d vector of size 'shape(sample_data)[axis] assigning each sample to one group/partition.
    :param sample_data: matrix containing samples for each distribution on axis 'axis'\n
        E.g. `(N, M)` matrix with `M` distributions containing `N` observed values each
    :param axis: the axis containing the data of one distribution
    :param optimizable: if true, the returned parameters will be optimizable
    :param name: name of the operation
    :return: negative binomial distribution
    """
    with tf.name_scope(name):
        uniques, partition_index = tf.unique(partitions, name="unique")
        
        def fit_part(i):
            smpl = tf.squeeze(tf.gather(sample_data, tf.where(tf.equal(partition_index, i)), axis=axis), axis=axis)
            w = None
            if weights is not None:
                w = tf.squeeze(tf.gather(weights, tf.where(tf.equal(partition_index, i)), axis=axis), axis=axis)
            dist = fit(smpl, axis=axis, weights=w, optimizable=optimizable,
                       validate_shape=validate_shape, dtype=dtype)
            
            retvalTuple = (
                dist.r,
                # dist.p,
                dist.mu
            )
            return retvalTuple
        
        idx = tf.range(tf.size(uniques))
        # r, p, mu = tf.map_fn(fit_part, idx, dtype=(sample_data.dtype, sample_data.dtype, sample_data.dtype))
        r, mu = tf.map_fn(fit_part, idx, dtype=(dtype, dtype))
        
        # shape(r) == shape(mu) == [ size(uniques), ..., 1, ... ]
        
        stacked_r = tf.gather(r, partition_index, name="r")
        # stacked_p = tf.gather(p, partition_index, name="p")
        stacked_mu = tf.gather(mu, partition_index, name="mu")
        
        # shape(r) == shape(mu) == [ shape(sample_data)[axis], ..., 1, ... ]
        
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
        
        # shape(r) == shape(mu) == shape(sample_data)
        
        return NegativeBinomial(r=stacked_r, mu=stacked_mu)


def fit(sample_data: tf.Tensor, axis=0, weights=None, optimizable=False,
        validate_shape=True,
        dtype=tf.float32,
        name="nb-dist") -> NegativeBinomial:
    """
    Fits negative binomial distributions NB(r, p) to given sample data along axis 'axis'.

    :param sample_data: matrix containing samples for each distribution on axis 'axis'\n
        E.g. `(N, M)` matrix with `M` distributions containing `N` observed values each
    :param axis: the axis containing the data of one distribution
    :param weights: if specified, the closed-form fit will be weighted
    :param optimizable: if true, the returned distribution's parameters will be optimizable
    :param name: A name for the returned distribution (optional).
    :return: negative binomial distribution
    """
    with tf.name_scope("fit"):
        (r, p) = fit_mme(sample_data, axis=axis, weights=weights)
        
        if optimizable:
            r = tf.Variable(name="r", initial_value=r, dtype=dtype, validate_shape=validate_shape)
            # r_var = tf.Variable(tf.zeros(tf.shape(r)), dtype=tf.float32, validate_shape=False, name="r_var")
            #
            # r_assign_op = tf.assign(r_var, r)
        
        # keep mu constant
        mu = reduce_weighted_mean(sample_data, weight=weights, axis=axis, keepdims=True, name="mu")
        
        distribution = NegativeBinomial(r=r, mu=mu, name=name)
    
    return distribution


def fit_mme(sample_data: tf.Tensor, axis=0, weights=None, replace_values=None, dtype=tf.float32,
            name="MME") -> (tf.Tensor, tf.Tensor):
    """
        Calculates the Maximum-of-Momentum Estimator of `NB(r, p)` for given sample data along axis 'axis.

        :param sample_data: matrix containing samples for each distribution on axis 'axis\n
            E.g. `(N, M)` matrix with `M` distributions containing `N` observed values each
        :param axis: the axis containing the data of one distribution
        :param weights: if specified, the fit will be weighted
        :param replace_values: Matrix of size `shape(sample_data)[1:]`
        :param dtype: data type of replacement values, if `replace_values` is not set;
            If None, the replacement values will be of the same data type like `sample_data`
        :param name: A name for the operation (optional).
        :return: estimated values of `r` and `p`
        """
    
    with tf.name_scope(name):
        mean = reduce_weighted_mean(sample_data, weight=weights, axis=axis, keepdims=True, name="mean")
        variance = reduce_weighted_mean(tf.square(sample_data - mean),
                                        weight=weights,
                                        axis=axis,
                                        keepdims=True,
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
