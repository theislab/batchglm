import abc

import tensorflow as tf

import impl.tf.util as tf_utils
from .external import AbstractEstimator, TFEstimator, TFEstimatorGraph, nb_utils


class EstimatorGraph(TFEstimatorGraph):
    sample_data: tf.Tensor

    r: tf.Tensor
    p: tf.Tensor
    mu: tf.Tensor
    mixture_assignment: tf.Tensor

    def __init__(self, num_mixtures, num_samples, num_distributions, graph=None, optimizable_nb=False):
        super().__init__(graph)

        # initial graph elements
        with self.graph.as_default():
            sample_data = tf.placeholder(tf.float32, shape=(num_samples, num_distributions), name="sample_data")
            initial_mixture_probs = tf.placeholder(tf.float32,
                                                   shape=(num_mixtures, num_samples),
                                                   name="initial_mixture_probs")

            with tf.name_scope("prepare_data"):
                # apply a random intercept to avoid zero gradients and infinite values
                with tf.name_scope("randomize"):
                    initial_mixture_probs += tf.random_uniform(initial_mixture_probs.shape, 0, 0.1,
                                                               dtype=tf.float32)
                    initial_mixture_probs = initial_mixture_probs / tf.reduce_sum(initial_mixture_probs, axis=0,
                                                                                  keepdims=True)
                    initial_mixture_probs = tf.identity(initial_mixture_probs, name="adjusted_initial_mixture_probs")

                with tf.name_scope("broadcast"):
                    sample_data = tf.expand_dims(sample_data, axis=0)
                    sample_data = tf.tile(sample_data, (num_mixtures, 1, 1))

            with tf.name_scope("mixture_prob"):
                # optimize logits to keep `mixture_prob` between the interval [0, 1]
                logit_mixture_prob = tf.Variable(tf_utils.logit(initial_mixture_probs),
                                                 name="logit_prob",
                                                 validate_shape=False)
                mixture_prob = tf.sigmoid(logit_mixture_prob, name="prob")
                # normalize: the assignment probabilities should sum up to 1
                # => `sum(mixture_prob of one sample) = 1`
                mixture_prob = tf.identity(mixture_prob / tf.reduce_sum(mixture_prob, axis=0, keepdims=True),
                                           name="normalize")
                mixture_prob = tf.expand_dims(mixture_prob, axis=-1)

            distribution = nb_utils.fit(sample_data=sample_data,
                                        axis=-2,
                                        weights=mixture_prob,
                                        name="fit_nb-dist")

            with tf.name_scope("count_probs"):
                with tf.name_scope("probs"):
                    probs = distribution.prob(sample_data)
                    # sum up: for k in num_mixtures: mixture_prob(k) * P(r_k, mu_k, sample_data)
                    probs = tf_utils.reduce_weighted_mean(probs, weight=mixture_prob, axis=-3)

                log_probs = tf.log(probs, name="log_probs")

            with tf.name_scope("training"):
                # minimize negative log probability (log(1) = 0)
                loss = -tf.reduce_sum(log_probs, name="loss")

                train_op = None
                # define train function
                if optimizable_nb:
                    train_op = tf.train.AdamOptimizer(learning_rate=0.05)
                    train_op = train_op.minimize(loss, global_step=tf.train.get_global_step())

            initializer_op = tf.global_variables_initializer()

            # parameters
            with tf.name_scope("mu"):
                mu = tf.reduce_sum(distribution.mu * mixture_prob, axis=-3)
            with tf.name_scope("r"):
                r = tf.reduce_sum(distribution.r * mixture_prob, axis=-3)
            with tf.name_scope("p"):
                p = tf.reduce_sum(distribution.p * mixture_prob, axis=-3)
            log_mu = tf.log(mu, name="log_mu")
            log_r = tf.log(r, name="log_r")
            log_p = tf.log(p, name="log_p")

            # set up class attributes
            self.sample_data = sample_data

            self.initializer_op = None

            self.r = r
            self.p = p
            self.mu = mu
            self.log_r = log_r
            self.log_p = log_p
            self.log_mu = log_mu

            self.distribution = distribution
            self.log_probs = log_probs

            self.loss = loss
            self.train_op = train_op

    def initialize(self, session, feed_dict, **kwargs):
        session.run(self.initializer_op, feed_dict=feed_dict)

    def train(self, session, feed_dict, *args, steps=1000, **kwargs):
        errors = []
        for i in range(steps):
            (loss_res, _) = session.run((self.loss, self.train_op),
                                        feed_dict=feed_dict)
            errors.append(loss_res)
            print(i)

        return errors


# g = EstimatorGraph(2, 2000, 10000, optimizable_nb=False)
# writer = tf.summary.FileWriter("/tmp/log/...", g.graph)
# writer.close()


class Estimator(AbstractEstimator, TFEstimator, metaclass=abc.ABCMeta):
    model: EstimatorGraph

    def __init__(self, input_data: dict, tf_estimator_graph=None):
        if tf_estimator_graph is None:
            num_mixtures = input_data["initial_mixture_probs"].shape[0]
            (num_samples, num_distributions) = input_data["sample_data"].shape
            tf_estimator_graph = EstimatorGraph(num_mixtures, num_samples, num_distributions)

        TFEstimator.__init__(self, input_data, tf_estimator_graph)

    @property
    def loss(self):
        return self.run(self.model.loss)

    @property
    def r(self):
        return self.run(self.model.r)

    @property
    def p(self):
        return self.run(self.model.p)

    @property
    def mu(self):
        return self.run(self.model.mu)

    @property
    @abc.abstractmethod
    def mixture_assignment(self):
        return self.run(self.model.mixture_assignment)
