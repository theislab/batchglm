from typing import Union

import tensorflow as tf

from impl.tf.ops import logit
from .external import nb_utils, tf_linreg


class MixtureModel:
    initializer: Union[tf.Tensor, tf.Operation]
    prob: tf.Tensor
    logit_prob: tf.Tensor
    
    def __init__(self, initial_values, name="mixture_prob"):
        with tf.name_scope(name):
            # optimize logits to keep `mixture_prob` between the interval [0, 1]
            logit_prob = tf.Variable(logit(initial_values), name="logit_prob")
            prob = tf.sigmoid(logit_prob.initialized_value(), name="prob")
            
            # normalize: the assignment probabilities should sum up to 1
            # => `sum(mixture_prob of one sample) = 1`
            with tf.name_scope("normalize"):
                prob = prob / tf.reduce_sum(prob, axis=0, keepdims=True)
            
            self.initializer = logit_prob.initializer
            self.prob = prob
            self.log_prob = tf.log(prob)
            self.logit_prob = logit_prob
        # assert (mixture_prob.shape == (num_mixtures, num_samples, 1))


class LinearBatchModel:
    a: tf.Tensor
    a_intercept: tf.Variable
    a_slope: tf.Variable
    b: tf.Tensor
    b_intercept: tf.Variable
    b_slope: tf.Variable
    log_mu_obs: tf.Tensor
    log_r_obs: tf.Tensor
    log_count_probs: tf.Tensor
    joint_log_probs: tf.Tensor
    loss: tf.Tensor
    
    def __init__(self,
                 init_dist: nb_utils.NegativeBinomial,
                 sample_index,
                 sample_data,
                 design,
                 mixture_model: MixtureModel,
                 name="Linear_Batch_Model"):
        with tf.name_scope(name):
            num_mixtures = mixture_model.prob.shape[0]
            num_design_params = design.shape[-1]
            (batch_size, num_genes) = sample_data.shape
            
            init_a_intercept = tf.log(init_dist.mean())
            init_b_intercept = tf.log(init_dist.r)
            
            mixture_log_prob = tf.gather(mixture_model.log_prob, sample_index, axis=-1)
            
            assert sample_data.shape == [batch_size, num_genes]
            assert design.shape == [batch_size, num_design_params]
            assert mixture_log_prob.shape == [num_mixtures, batch_size]
            assert init_a_intercept.shape == [num_mixtures, 1, num_genes] == init_b_intercept.shape
            
            init_a_slopes = tf.log(tf.random_uniform(
                tf.TensorShape([1, num_design_params - 1, num_genes]),
                maxval=0.1,
                dtype=design.dtype
            ))
            
            init_b_slopes = init_a_slopes
            
            a, a_intercept, a_slope = tf_linreg.param_variable(init_a_intercept, init_a_slopes, name="a")
            b, b_intercept, b_slope = tf_linreg.param_variable(init_b_intercept, init_b_slopes, name="b")
            assert a.shape == (num_mixtures, num_design_params, num_genes) == b.shape
            
            dist_estim = nb_utils.NegativeBinomial(mean=tf.exp(a_intercept),
                                                   r=tf.exp(b_intercept),
                                                   name="dist_estim")
            
            with tf.name_scope("broadcast"):
                design = tf.expand_dims(design, axis=0)
            design = tf.tile(design, (num_mixtures, 1, 1))
            assert (design.shape == (num_mixtures, batch_size, num_design_params))
            
            with tf.name_scope("mu"):
                log_mu = tf.matmul(design, a, name="log_mu_obs")
                log_mu = tf.clip_by_value(log_mu, log_mu.dtype.min, log_mu.dtype.max)
                mu = tf.exp(log_mu)
            
            with tf.name_scope("r"):
                log_r = tf.matmul(design, b, name="log_r_obs")
                log_r = tf.clip_by_value(log_r, log_r.dtype.min, log_r.dtype.max)
                r = tf.exp(log_r)
            
            dist_obs = nb_utils.NegativeBinomial(r=r, mean=mu, name="dist_obs")
            
            # calculate mixture model probability:
            log_count_probs = dist_obs.log_prob(tf.expand_dims(sample_data, 0), name="log_count_probs")
            
            # sum up: for k in num_mixtures: mixture_prob(k) * P(r_k, mu_k, sample_data)
            joint_log_probs = tf.reduce_logsumexp(log_count_probs + tf.expand_dims(mixture_log_prob, -1),
                                                  axis=-3,
                                                  name="joint_log_probs")
            
            # probs = tf.exp(joint_log_probs, name="joint_probs")
            
            # minimize negative log probability (log(1) = 0);
            # use the mean loss to keep a constant learning rate independently of the batch size
            loss = -tf.reduce_mean(joint_log_probs, name="loss")
            
            self.a = a
            self.a_intercept = a_intercept
            self.a_slope = a_slope
            self.b = b
            self.b_intercept = b_intercept
            self.b_slope = b_slope
            self.dist_estim = dist_estim
            self.dist_obs = dist_obs
            self.log_mu_obs = log_mu
            self.log_r_obs = log_r
            self.log_count_probs = log_count_probs
            self.joint_log_probs = joint_log_probs
            self.loss = loss
