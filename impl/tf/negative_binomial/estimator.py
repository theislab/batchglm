import abc

import xarray as xr

import tensorflow as tf

import impl.tf.ops as tf_ops
from .external import AbstractEstimator, TFEstimator, TFEstimatorGraph
from .util import fit, NegativeBinomial

PARAMS = {
    "mu": ("samples", "genes"),
    "sigma2": ("samples", "genes"),
    "loss": ()
}


class EstimatorGraph(TFEstimatorGraph):
    sample_data: tf.Tensor

    mu: tf.Tensor
    sigma2: tf.Tensor

    def __init__(self, num_samples, num_genes, graph=None, optimizable=True):
        super().__init__(graph)

        # initial graph elements
        with self.graph.as_default():
            sample_data = tf_ops.caching_placeholder(tf.float32, shape=(num_samples, num_genes), name="sample_data")

            learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
            global_train_step = tf.Variable(0, name="train_step")

            distribution = fit(sample_data=sample_data, optimizable=optimizable, name="fit_nb-dist")

            count_probs = distribution.prob(sample_data, name="count_probs")
            log_count_probs = distribution.log_prob(sample_data, name="log_count_probs")

            with tf.name_scope("training"):
                # minimize negative log probability (log(1) = 0)
                log_likelihood = tf.reduce_sum(log_count_probs, name="log_likelihood")
                with tf.name_scope("loss"):
                    loss = -tf.reduce_mean(log_count_probs)

                # define train function
                optimizer = None
                gradient = None
                train_op = None
                if optimizable:
                    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                    gradient = optimizer.compute_gradients(loss)
                    train_op = optimizer.minimize(loss, global_step=global_train_step)

            # set up class attributes
            self.sample_data = sample_data

            self.initializer_op = tf.global_variables_initializer()

            self.mu = distribution.mean()
            self.r = distribution.r
            self.sigma2 = distribution.variance()

            self.distribution = distribution
            self.count_probs = count_probs
            self.log_count_probs = log_count_probs
            self.log_likelihood = log_likelihood

            self.loss = loss
            self.optimizer = optimizer
            self.gradient = gradient
            self.train_op = train_op
            self.global_train_step = global_train_step

            self.initializer_ops = [
                tf.variables_initializer([sample_data]),
                tf.variables_initializer(tf.global_variables()),
            ]

    def initialize(self, session, feed_dict, **kwargs):
        with self.graph.as_default():
            for op in self.initializer_ops:
                session.run(op, feed_dict=feed_dict)

    def train(self, session, feed_dict, *args, steps=1000, learning_rate=0.05, **kwargs):
        print("learning rate: %s" % learning_rate)
        errors = []
        for i in range(steps):
            # feed_dict = feed_dict.copy()
            feed_dict = dict()
            feed_dict["learning_rate:0"] = learning_rate
            (train_step, loss_res, _) = session.run((self.global_train_step, self.loss, self.train_op),
                                                    feed_dict=feed_dict)
            errors.append(loss_res)
            print("Step: %d\tloss: %f" % (train_step, loss_res))

        return errors


class Estimator(AbstractEstimator, TFEstimator, metaclass=abc.ABCMeta):
    model: EstimatorGraph

    def __init__(self, input_data: xr.Dataset, model=None, optimizable=False):
        num_genes = input_data.dims["genes"]
        num_samples = input_data.dims["samples"]

        if model is None:
            tf.reset_default_graph()

            model = EstimatorGraph(num_samples, num_genes,
                                   optimizable=optimizable,
                                   graph=tf.get_default_graph())

        TFEstimator.__init__(self, input_data, model)

    @property
    def loss(self):
        return self.run(self.model.loss)

    @property
    def mu(self):
        return self.run(self.model.mu)

    @property
    def r(self):
        return self.run(self.model.r)

    @property
    def sigma2(self):
        return self.run(self.model.sigma2)

    @property
    def count_probs(self):
        return self.run(self.model.count_probs)

    @property
    def log_count_probs(self):
        return self.run(self.model.log_count_probs)

    @property
    def log_likelihood(self):
        return self.run(self.model.log_likelihood)
