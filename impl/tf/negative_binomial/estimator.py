import abc

import tensorflow as tf

from .external import AbstractEstimator, TFEstimator, TFEstimatorGraph
from .util import fit


class EstimatorGraph(TFEstimatorGraph):
    sample_data: tf.Tensor

    r: tf.Tensor
    p: tf.Tensor
    mu: tf.Tensor

    def __init__(self, graph=None, optimizable_nb=True):
        super().__init__(graph)

        # initial graph elements
        with self.graph.as_default():
            sample_data = tf.placeholder(tf.float32, name="sample_data")

            distribution = fit(sample_data=sample_data, optimizable=optimizable_nb,
                               validate_shape=False, name="fit_nb-dist")
            log_probs = tf.identity(distribution.log_prob(sample_data), name="log_probs")

            with tf.name_scope("training"):
                # minimize negative log probability (log(1) = 0)
                loss = -tf.reduce_sum(log_probs, name="loss")

                train_op = None
                # define train function
                if optimizable_nb:
                    optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
                    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            initializer_op = tf.global_variables_initializer()

            # parameters
            mu = distribution.mu
            mu = tf.identity(mu, name="mu")
            r = distribution.r
            r = tf.identity(r, name="r")
            p = distribution.p
            p = tf.identity(p, name="p")
            log_r = tf.log(r, name="log_r")
            log_p = tf.log(p, name="log_p")
            log_mu = tf.log(mu, name="log_mu")

            # set up class attributes
            self.sample_data = sample_data

            self.initializer_op = tf.global_variables_initializer()

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


# g = EstimatorGraph(optimizable_nb=False)
# writer = tf.summary.FileWriter("/tmp/log/...", g.graph)


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
