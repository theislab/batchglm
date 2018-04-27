import abc
from typing import Union

import tensorflow as tf
import numpy as np

import impl.tf.util as tf_utils
from .external import AbstractEstimator, TFEstimator, TFEstimatorGraph
from .external import nb_utils, tf_linreg
from tensorflow.contrib.distributions import Distribution


class MixtureModel:
    initializer: Union[tf.Tensor, tf.Operation]
    prob: tf.Tensor
    logit_prob: tf.Tensor

    def __init__(self, initial_values, name="mixture_prob"):
        with tf.name_scope(name):
            # optimize logits to keep `mixture_prob` between the interval [0, 1]
            logit_prob = tf.Variable(tf_utils.logit(initial_values), name="logit_prob")
            prob = tf.sigmoid(logit_prob, name="prob")

            # normalize: the assignment probabilities should sum up to 1
            # => `sum(mixture_prob of one sample) = 1`
            with tf.name_scope("normalize"):
                prob = prob / tf.reduce_sum(prob, axis=0, keepdims=True)

            self.initializer = logit_prob.initializer
            self.prob = prob
            self.logit_prob = logit_prob
        # assert (mixture_prob.shape == (num_mixtures, num_samples, 1))


class LinearBatchModel:
    initializer: Union[tf.Tensor, tf.Operation]
    a: tf.Tensor
    b: tf.Tensor
    loss: tf.Tensor

    def __init__(self,
                 init_a_intercept: tf.Tensor,
                 init_b_intercept: tf.Tensor,
                 sample_data, design, mixture_prob,
                 name="Linear_Batch_Model"):
        with tf.name_scope(name):
            num_mixtures = init_a_intercept.shape[0]
            num_design_params = design.shape[-1]
            (batch_size, num_genes) = sample_data.shape

            assert sample_data.shape == [batch_size, num_genes]
            assert design.shape == [batch_size, num_design_params]
            assert mixture_prob.shape == [num_mixtures, batch_size]
            assert init_a_intercept.shape == [num_mixtures, 1, num_genes] == init_b_intercept.shape

            init_a_slopes = tf.truncated_normal(tf.TensorShape([1, num_design_params - 1, num_genes]),
                                                dtype=design.dtype)

            init_b_slopes = tf.truncated_normal(tf.TensorShape([1, num_design_params - 1, num_genes]),
                                                dtype=design.dtype)

            a = tf_linreg.param_variable(init_a_intercept, init_a_slopes, name="a")
            b = tf_linreg.param_variable(init_b_intercept, init_b_slopes, name="b")
            assert a.shape == (num_mixtures, num_design_params, num_genes) == b.shape

            with tf.name_scope("broadcast"):
                design = tf.expand_dims(design, axis=0)
                design = tf.tile(design, (num_mixtures, 1, 1))
                assert (design.shape == (num_mixtures, batch_size, num_design_params))

            log_mu_obs = tf.matmul(design, a, name="log_mu_obs")
            log_sigma2_obs = tf.matmul(design, b, name="log_sigma2_obs")

            # calculate mixture model probability:
            dist_obs = nb_utils.NegativeBinomial(mean=tf.exp(log_mu_obs),
                                                 variance=tf.exp(log_sigma2_obs),
                                                 name="NB_dist")

            count_probs = dist_obs.prob(sample_data, name="count_probs")
            # sum up: for k in num_mixtures: mixture_prob(k) * P(r_k, mu_k, sample_data)
            joint_probs = tf_utils.reduce_weighted_mean(count_probs,
                                                        weight=tf.expand_dims(mixture_prob, -1),
                                                        axis=-3,
                                                        name="joint_probs")

            log_probs = tf.log(joint_probs, name="log_probs")

            # minimize negative log probability (log(1) = 0);
            # use the mean loss to keep a constant learning rate independently of the batch size
            loss = -tf.reduce_mean(log_probs, name="loss")

        self.a = a
        self.b = b
        self.loss = loss
        self.initializer = tf.group(a.initializer, b.initializer)


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

    initializers: dict

    def __init__(
            self,
            sample_data,
            design,
            initial_mixture_probs,
            # num_mixtures,
            # num_samples,
            # num_genes,
            # num_design_params,
            graph=None
    ):
        super().__init__(graph)
        # initial graph elements
        with self.graph.as_default():
            # shape parameters
            num_mixtures = initial_mixture_probs.shape[0]
            num_design_params = design.shape[-1]
            (num_samples, num_genes) = sample_data.shape

            initializers = []

            # data preparation
            # broadcast data
            # TODO: change tf.tile to tf.broadcast_to after next TF release
            with tf.name_scope("broadcast"):
                initial_mixture_probs = tf.expand_dims(initial_mixture_probs, -1)
                assert (initial_mixture_probs.shape == (num_mixtures, num_samples, 1))

                sample_data = tf.expand_dims(sample_data, axis=0)
                sample_data = tf.tile(sample_data, (num_mixtures, 1, 1))
                assert (sample_data.shape == (num_mixtures, num_samples, num_genes))

                design = tf.expand_dims(design, axis=0)
                design = tf.tile(design, (num_mixtures, 1, 1))
                assert (design.shape == (num_mixtures, num_samples, num_design_params))

            with tf.name_scope("initialization"):
                # implicit broadcasting of sample_data and initial_mixture_probs to
                # shape (num_mixtures, num_samples, num_genes)
                init_dist = nb_utils.fit(tf.expand_dims(sample_data, 0),
                                         weights=tf.expand_dims(initial_mixture_probs, -1), axis=-2)
                assert init_dist.r.shape == [num_mixtures, 1, num_genes]

                init_a_intercept = tf.log(init_dist.mean())
                init_b_intercept = tf.log(init_dist.variance())

            # define variables
            # confounder vector is batch-independent
            mu = tf.Variable(init_dist.mean(), name="mu")
            sigma2 = tf.Variable(init_dist.variance(), name="var")

            batch_model = LinearBatchModel(init_a_intercept, init_b_intercept, batch_)

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

            self.initializer_op = None
            self.train_op = None

    def initialize(self, session, feed_dict, **kwargs):
        with self.graph.as_default():
            if self.initializer_op is not None:
                session.run(self.initializer_op, feed_dict=feed_dict)
            else:
                session.run(tf.global_variables_initializer(), feed_dict=feed_dict)

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

    def __init__(self, input_data: dict,
                 batch_size=250,
                 random_effect=0.1,
                 model=None):
        if model is None:
            g = tf.Graph()
            with g.as_default():
                # shape parameters
                num_mixtures = input_data["initial_mixture_probs"].shape[0]
                num_design_params = input_data["design"].shape[-1]
                (num_samples, num_genes) = input_data["sample_data"].shape

                # num_mixtures = tf.convert_to_tensor(num_mixtures, name="num_mixtures")
                # num_design_params = tf.convert_to_tensor(num_design_params, name="num_design_params")
                # num_samples = tf.convert_to_tensor(num_samples, name="num_samples")
                # num_genes = tf.convert_to_tensor(num_genes, name="num_genes")

                # placeholders
                sample_data = tf.placeholder(tf.float32, shape=(num_samples, num_genes), name="sample_data")
                design = tf.placeholder(tf.float32, shape=(num_samples, num_design_params), name="design")
                initial_mixture_probs = tf.placeholder(tf.float32,
                                                       shape=(num_mixtures, num_samples),
                                                       name="initial_mixture_probs")

                learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")

                # apply a random intercept to avoid zero gradients and infinite values
                with tf.name_scope("randomize"):
                    initial_mixture_probs += tf.random_uniform(initial_mixture_probs.shape, 0, random_effect,
                                                               dtype=tf.float32)
                    initial_mixture_probs = initial_mixture_probs / tf.reduce_sum(initial_mixture_probs, axis=0,
                                                                                  keepdims=True)

                # define mixture parameters
                mixture_model = MixtureModel(initial_mixture_probs)
                mixture_probs = tf.transpose(mixture_model.prob)

                # define mean and variance

                data = tf.data.Dataset.from_tensor_slices((
                    sample_data,
                    design,

                    mixture_probs
                ))
                data = data.repeat()
                data = data.shuffle(batch_size * 4)
                data = data.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

                iterator = data.make_initializable_iterator()

                batch_sample_data, batch_design, batch_mixture_prob = iterator.get_next()
                batch_mixture_prob = tf.transpose(batch_mixture_prob)

                model = EstimatorGraph(sample_data, design, initial_mixture_probs,
                                       # num_mixtures, num_samples, num_genes, num_design_params,
                                       graph=g)

                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                train_op = optimizer.minimize(model.loss, global_step=tf.train.get_global_step())

                model.train_op = train_op

        TFEstimator.__init__(self, input_data, model)

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
