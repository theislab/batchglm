import abc

import tensorflow as tf

from . import AbstractEstimator, TFEstimator, TFEstimatorGraph, fit_partitioned_nb, linear_regression, negative_binomial

__all__ = ['EstimatorGraph', 'Estimator']


class EstimatorGraph(TFEstimatorGraph):
    sample_data: tf.Tensor

    r: tf.Tensor
    p: tf.Tensor
    mu: tf.Tensor
    a: tf.Tensor

    def __init__(self, graph=tf.Graph(), trainable_nb=False, trainable_lm=False):
        super().__init__(graph)

        # initial graph elements
        with self.graph.as_default():
            self.sample_data = tf.placeholder(tf.float32, name="sample_data")
            self.design = tf.placeholder(tf.int32, name="design")

            self.mu = tf.reduce_mean(self.sample_data, axis=0, name="mu")
            (self.r, self.p) = fit_partitioned_nb(self.sample_data, self.design)

            self.a, loss = linear_regression(self.design, tf.log(self.r), name="lin_reg_a")
            tf.identity(self.a, "a")
            self.b, loss = linear_regression(self.design, tf.log(self.mu), name="lin_reg_b")
            tf.identity(self.b, "b")

            self.probs = self.distribution.log_prob(self.sample_data)
            self.distribution = negative_binomial(r=tf.exp(self.a[0]), mu=tf.exp(self.b[0]))

            # # minimize negative log probability (log(1) = 0)
            # self.loss = -tf.reduce_sum(self.probs, name="loss")
            #
            # # define train function
            # self.train_op = tf.train.AdamOptimizer(learning_rate=0.05)
            # self.train_op = self.train_op.minimize(self.loss, global_step=tf.train.get_global_step())
            #
            # self.initializer_op = tf.global_variables_initializer()
            #
            # # parameters
            # self.mu = tf.reduce_mean(self.sample_data, axis=0, name="mu")
            # self.r = self.distribution.total_count
            # self.p = self.distribution.prob

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


class Estimator(AbstractEstimator, TFEstimator, metaclass=abc.ABCMeta):
    model: EstimatorGraph

    def __init__(self, input_data: dict, tf_estimator_graph=None):
        if tf_estimator_graph is None:
            tf_estimator_graph = EstimatorGraph()

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
