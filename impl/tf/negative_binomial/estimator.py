import abc

import xarray as xr
import numpy as np
import tensorflow as tf

try:
    import anndata
except ImportError:
    anndata = None

from .external import AbstractEstimator, MonitoredTFEstimator, TFEstimatorGraph
from . import util as nb_utils

ESTIMATOR_PARAMS = AbstractEstimator.params().copy()
ESTIMATOR_PARAMS.update({
    "mu_raw": ("genes",),
    "r_raw": ("genes",),
    "sigma2_raw": ("genes",),
})


class EstimatorGraph(TFEstimatorGraph):
    sample_data: tf.Tensor

    mu: tf.Tensor
    r: tf.Tensor
    sigma2: tf.Tensor

    def __init__(self, sample_data, num_samples, num_genes, graph=None, optimizable=True):
        super().__init__(graph)

        # initial graph elements
        with self.graph.as_default():
            # # does not work due to broken tf.Variable initialization.
            # # Integrate data directly into the graph, until this issue is fixed.
            # sample_data_var = tf_ops.caching_placeholder(
            #     tf.float32,
            #     shape=(num_samples, num_genes), name="sample_data")
            # sample_data = sample_data_var.initialized_value()

            learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
            global_step = tf.train.get_or_create_global_step()

            distribution = nb_utils.fit(sample_data=sample_data, optimizable=optimizable, name="fit_nb-dist")

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
                    train_op = optimizer.minimize(loss, global_step=global_step)

            # set up class attributes
            self.sample_data = sample_data

            self.mu_raw = distribution.mean()
            self.r_raw = distribution.r
            self.sigma2_raw = distribution.variance()

            self.mu = tf.tile(self.mu_raw, (num_samples, 1))
            self.r = tf.tile(self.r_raw, (num_samples, 1))
            self.sigma2 = tf.tile(self.sigma2_raw, (num_samples, 1))

            self.distribution = distribution
            self.count_probs = count_probs
            self.log_count_probs = log_count_probs
            self.log_likelihood = log_likelihood

            self.optimizer = optimizer
            self.gradient = gradient

            self.loss = loss
            self.train_op = train_op
            self.init_op = tf.global_variables_initializer()
            self.global_step = global_step


class Estimator(AbstractEstimator, MonitoredTFEstimator, metaclass=abc.ABCMeta):
    model: EstimatorGraph

    @classmethod
    def params(cls) -> dict:
        return ESTIMATOR_PARAMS

    def __init__(self, input_data: xr.Dataset, model=None, fast=False):
        if model is None:
            tf.reset_default_graph()

            # read input_data
            if anndata is not None and isinstance(input_data, anndata.AnnData):
                # TODO: load sample_data as sparse array instead of casting it to a dense numpy array
                sample_data = np.asarray(input_data.X, dtype=np.float32)

                num_genes = input_data.n_vars
                num_samples = input_data.n_obs
            elif isinstance(input_data, xr.Dataset):
                sample_data = np.asarray(input_data["sample_data"], dtype=np.float32)

                num_genes = input_data.dims["genes"]
                num_samples = input_data.dims["samples"]
            else:
                sample_data = input_data
                (num_samples, num_genes) = input_data.shape

            self._sample_data = sample_data

            model = EstimatorGraph(sample_data,
                                   num_samples, num_genes,
                                   optimizable=not fast,
                                   graph=tf.get_default_graph())

        MonitoredTFEstimator.__init__(self, input_data, model)

    def _scaffold(self):
        with self.model.graph.as_default():
            scaffold = tf.train.Scaffold(
                init_op=self.model.init_op,
            )
        return scaffold

    def train(self, *args, learning_rate=0.05, **kwargs):
        tf.logging.info("learning rate: %s" % learning_rate)
        super().train(feed_dict={"learning_rate:0": learning_rate})

    @property
    def sample_data(self):
        return self._sample_data

    @property
    def mu(self):
        return self.get("mu")

    @property
    def r(self):
        return self.get("r")

    @property
    def sigma2(self):
        return self.get("sigma2")

    def count_probs(self, sample_data=None):
        if sample_data is None:
            return self.get("count_probs")
        else:
            return super().count_probs(sample_data)

    def log_count_probs(self, sample_data=None):
        if sample_data is None:
            return self.get("log_count_probs")
        else:
            return super().log_count_probs(sample_data)

    def log_likelihood(self, sample_data=None):
        if sample_data is None:
            return self.get("log_likelihood")
        else:
            return super().log_likelihood(sample_data)
