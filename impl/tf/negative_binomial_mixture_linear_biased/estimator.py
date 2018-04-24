import abc

import tensorflow as tf
import numpy as np

import impl.tf.util as tf_utils
from .external import AbstractEstimator, TFEstimator, TFEstimatorGraph
from .external import nb_utils, LinearRegression
from tensorflow.contrib.distributions import Distribution


class EstimatorGraph(TFEstimatorGraph):
    sample_data: tf.Tensor
    design: tf.Tensor

    dist_obs: Distribution
    dist_estim: Distribution

    mu: tf.Tensor
    sigma2: tf.Tensor
    a: tf.Tensor
    b: tf.Tensor
    mixture_prob: tf.Tensor
    mixture_assignment: tf.Tensor

    def __init__(
            self,
            sample_data,
            design,
            initial_mixture_probs,
            learning_rate,
            num_mixtures,
            num_samples,
            num_genes,
            num_design_params,
            graph=None,
            random_effect=0.1
    ):
        super().__init__(graph)
        # initial graph elements
        with self.graph.as_default():
            # data preparation
            with tf.name_scope("prepare_data"):
                # apply a random intercept to avoid zero gradients and infinite values
                with tf.name_scope("randomize"):
                    initial_mixture_probs += tf.random_uniform(initial_mixture_probs.shape, 0, random_effect,
                                                               dtype=tf.float32)
                    initial_mixture_probs = initial_mixture_probs / tf.reduce_sum(initial_mixture_probs, axis=0,
                                                                                  keepdims=True)
                    initial_mixture_probs = tf.expand_dims(initial_mixture_probs, -1)
                    initial_mixture_probs = tf.identity(initial_mixture_probs, name="adjusted_initial_mixture_probs")
                assert (initial_mixture_probs.shape == (num_mixtures, num_samples, 1))

                # broadcast sample data to shape (num_mixtures, num_samples, num_genes)
                with tf.name_scope("broadcast"):
                    sample_data = tf.expand_dims(sample_data, axis=0)
                    sample_data = tf.tile(sample_data, (num_mixtures, 1, 1))
                    # TODO: change tf.tile to tf.broadcast_to after next TF release
                assert (sample_data.shape == (num_mixtures, num_samples, num_genes))

                # broadcast sample data to shape (num_mixtures, num_samples, num_design_params)
                with tf.name_scope("broadcast"):
                    design = tf.expand_dims(design, axis=0)
                    design = tf.tile(design, (num_mixtures, 1, 1))
                    # TODO: change tf.tile to tf.broadcast_to after next TF release
                assert (design.shape == (num_mixtures, num_samples, num_design_params))

            # define mixture_prob tensor depending on optimization method
            mixture_prob = None
            with tf.name_scope("mixture_prob"):
                # optimize logits to keep `mixture_prob` between the interval [0, 1]
                logit_mixture_prob = tf.Variable(tf_utils.logit(initial_mixture_probs), name="logit_prob")
                mixture_prob = tf.sigmoid(logit_mixture_prob, name="prob")

                # normalize: the assignment probabilities should sum up to 1
                # => `sum(mixture_prob of one sample) = 1`
                mixture_prob = mixture_prob / tf.reduce_sum(mixture_prob, axis=0, keepdims=True)
                mixture_prob = tf.identity(mixture_prob, name="normalize")
            assert (mixture_prob.shape == (num_mixtures, num_samples, 1))

            with tf.name_scope("initialization"):
                init_dist = nb_utils.fit(sample_data, weights=initial_mixture_probs, axis=-2)

                init_a_intercept = tf.log(init_dist.mean())
                init_a_slopes = tf.truncated_normal([1, num_design_params - 1, num_genes],
                                                    dtype=design.dtype)

                init_b_intercept = tf.log(init_dist.variance())
                init_b_slopes = tf.truncated_normal([1, num_design_params - 1, num_genes],
                                                    dtype=design.dtype)

            # define variables
            mu_obs = tf.Variable(tf.tile(init_dist.mean(), (1, num_samples, 1)), name="mu")
            sigma2_obs = tf.Variable(tf.tile(init_dist.variance(), (1, num_samples, 1)), name="var")

            log_mu_obs = tf.log(mu_obs, name="log_mu_obs")
            log_sigma2_obs = tf.log(sigma2_obs, name="log_var_obs")

            with tf.name_scope("variable_a"):
                a_intercept = tf.Variable(init_a_intercept, name='a_intercept')
                a_slopes = tf.Variable(init_a_slopes, name='a_slopes')
                a_slopes = tf.tile(a_slopes, (num_mixtures, 1, 1), name="constraint_a")
                a = tf.concat([
                    a_intercept,
                    a_slopes
                ], axis=-2)
                a = tf.identity(a, name="a")
            assert a.shape == (num_mixtures, num_design_params, num_genes)

            linreg_a = LinearRegression(design, log_sigma2_obs, b=a, name="lin_reg_a", fast=False)

            with tf.name_scope("variable_b"):
                b_intercept = tf.Variable(init_b_intercept, name='a_intercept')
                b_slopes = tf.Variable(init_b_slopes, name='a_slopes')
                b_slopes = tf.tile(b_slopes, (num_mixtures, 1, 1), name="constraint_a")
                b = tf.concat([
                    b_intercept,
                    b_slopes
                ], axis=-2)
                b = tf.identity(b, name="b")
            assert a.shape == (num_mixtures, num_design_params, num_genes)

            linreg_b = LinearRegression(design, log_mu_obs, b=b, name="lin_reg_b", fast=False)

            # calculate mixture model probability:
            dist_obs = nb_utils.NegativeBinomial(mean=mu_obs, variance=sigma2_obs, name="NB_dist")

            count_probs = dist_obs.prob(sample_data, name="count_probs")
            # sum up: for k in num_mixtures: mixture_prob(k) * P(r_k, mu_k, sample_data)
            joint_probs = tf_utils.reduce_weighted_mean(count_probs, weight=mixture_prob, axis=-3,
                                                        name="joint_probs")

            log_probs = tf.log(joint_probs, name="log_probs")

            # minimize negative log probability (log(1) = 0)
            loss = -tf.reduce_sum(log_probs, name="loss")
            # # define training operations
            # with tf.name_scope("training"):
            #     optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            #     train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            with tf.name_scope("var_estim"):
                var_estim = tf.exp(tf.gather(a, tf.zeros(num_samples, dtype=tf.int32), axis=-2))
                # var_estim = tf.exp(a[:, 0, :])
            with tf.name_scope("mu_estim"):
                mu_estim = tf.exp(tf.gather(b, tf.zeros(num_samples, dtype=tf.int32), axis=-2))
                # mu_estim = tf.exp(b[:, 0, :])
            dist_estim = nb_utils.NegativeBinomial(variance=var_estim, mean=mu_estim, name="dist_estim")

            # set up class attributes
            self.sample_data = sample_data
            self.design = design

            self.initializer_op = tf.global_variables_initializer()

            self.distribution_obs = dist_obs
            self.distribution_estim = dist_estim

            self.mu = tf.reduce_sum(dist_obs.mean() * mixture_prob, axis=-3)
            self.sigma2 = tf.reduce_sum(dist_obs.variance() * mixture_prob, axis=-3)
            self.log_mu = tf.log(self.mu, name="log_mu")
            self.log_sigma2 = tf.log(self.sigma2, name="log_sigma2")
            self.a = a
            self.b = b
            assert (self.mu.shape == (num_samples, num_genes))
            assert (self.sigma2.shape == (num_samples, num_genes))
            assert (self.a.shape == (num_mixtures, num_design_params, num_genes))
            assert (self.b.shape == (num_mixtures, num_design_params, num_genes))

            self.mixture_prob = tf.squeeze(mixture_prob)
            with tf.name_scope("mixture_assignment"):
                self.mixture_assignment = tf.squeeze(tf.argmax(mixture_prob, axis=0))
            assert (self.mixture_prob.shape == (num_mixtures, num_samples))
            assert (self.mixture_assignment.shape == num_samples)

            self.log_probs = log_probs
            self.loss = loss
            # self.train_op = train_op

    def initialize(self, session, feed_dict, **kwargs):
        session.run(self.initializer_op, feed_dict=feed_dict)

    def train(self, session, feed_dict, *args, steps=1000, learning_rate=0.05, **kwargs):
        if self.train_op is None:
            raise RuntimeWarning("this graph is not trainable")
        print(learning_rate)
        errors = []
        for i in range(steps):
            feed_dict = feed_dict.copy()
            feed_dict["learning_rate:0"] = learning_rate
            (loss_res, _) = session.run((self.loss, self.train_op),
                                        feed_dict=feed_dict)
            errors.append(loss_res)
            print(i)

        return errors


# g = EstimatorGraph(sim.data.design, optimizable_nb=False)
# writer = tf.summary.FileWriter("/tmp/log/...", g.graph)


class Estimator(AbstractEstimator, TFEstimator, metaclass=abc.ABCMeta):
    model: EstimatorGraph

    def __init__(self, input_data: dict, use_em=False, tf_estimator_graph=None):
        if tf_estimator_graph is None:
            # shape parameters
            num_mixtures = tf.convert_to_tensor(input_data["initial_mixture_probs"].shape[0], name="num_mixtures")
            num_design_params = tf.convert_to_tensor(input_data["design"].shape[-1], name="num_design_params")
            (num_samples, num_genes) = input_data["sample_data"].shape
            num_samples = tf.convert_to_tensor(num_samples, name="num_samples")
            num_genes = tf.convert_to_tensor(num_genes, name="num_genes")

            # placeholders
            sample_data = tf.placeholder(tf.float32, shape=(num_samples, num_genes), name="sample_data")
            design = tf.placeholder(tf.float32, shape=(num_samples, num_design_params), name="design")
            initial_mixture_probs = tf.placeholder(tf.float32,
                                                   shape=(num_mixtures, num_samples),
                                                   name="initial_mixture_probs")
            learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")

            tf_estimator_graph = EstimatorGraph(sample_data, design, initial_mixture_probs, learning_rate,
                                                num_mixtures, num_samples, num_genes, num_design_params)

        TFEstimator.__init__(self, input_data, tf_estimator_graph)

    @property
    def loss(self):
        return self.run(self.model.loss)

    @property
    def mu(self):
        return self.run(self.model.mu)

    @property
    def sigma2(self):
        return self.run(self.model.sigma2)

    @property
    def a(self):
        return self.run(self.model.a)

    @property
    def b(self):
        return self.run(self.model.b)

    @property
    def mixture_assignment(self):
        return self.run(self.model.mixture_assignment)

    @property
    def mixture_prob(self):
        return self.run(self.model.mixture_prob)
