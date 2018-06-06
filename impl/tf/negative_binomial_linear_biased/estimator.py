import abc
from typing import List, Union

import os
import datetime

import xarray as xr
import tensorflow as tf
import numpy as np

try:
    import anndata
except ImportError:
    anndata = None

# import impl.tf.ops as tf_ops
import impl.tf.util as tf_utils
from impl.tf.train import TimedRunHook
from .external import AbstractEstimator, TFEstimator, TFEstimatorGraph
from .external import nb_utils, tf_linreg

# session / device config
CONFIG = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

PARAMS = {
    "a": ("design_params", "genes"),
    "b": ("design_params", "genes"),
    "mu": ("samples", "genes"),
    "sigma2": ("samples", "genes"),
    "loss": ()
}


class LinearBatchModel:
    a: tf.Tensor
    a_intercept: tf.Variable
    a_slope: tf.Variable
    b: tf.Tensor
    b_intercept: tf.Variable
    b_slope: tf.Variable
    dist_estim: nb_utils.NegativeBinomial
    dist_obs: nb_utils.NegativeBinomial
    log_mu_obs: tf.Tensor
    log_r_obs: tf.Tensor
    log_count_probs: tf.Tensor
    joint_log_probs: tf.Tensor
    loss: tf.Tensor

    def __init__(self,
                 init_dist: nb_utils.NegativeBinomial,
                 sample_data,
                 design,
                 name="Linear_Batch_Model"):
        with tf.name_scope(name):
            num_design_params = design.shape[-1]
            (batch_size, num_genes) = sample_data.shape

            assert sample_data.shape == [batch_size, num_genes]
            assert design.shape == [batch_size, num_design_params]

            with tf.name_scope("initialization"):
                init_a_intercept = tf.log(init_dist.mean())
                init_b_intercept = tf.log(init_dist.r)

                assert init_a_intercept.shape == [1, num_genes] == init_b_intercept.shape

                init_a_slopes = tf.log(tf.random_uniform(
                    tf.TensorShape([num_design_params - 1, num_genes]),
                    maxval=0.1,
                    dtype=design.dtype
                ))

                init_b_slopes = init_a_slopes

            a, a_intercept, a_slope = tf_linreg.param_variable(init_a_intercept, init_a_slopes, name="a")
            b, b_intercept, b_slope = tf_linreg.param_variable(init_b_intercept, init_b_slopes, name="b")
            assert a.shape == (num_design_params, num_genes) == b.shape

            dist_estim = nb_utils.NegativeBinomial(mean=tf.exp(a_intercept),
                                                   r=tf.exp(b_intercept),
                                                   name="dist_estim")

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
            log_count_probs = dist_obs.log_prob(sample_data, name="log_count_probs")

            # minimize negative log probability (log(1) = 0);
            # use the mean loss to keep a constant learning rate independently of the batch size
            loss = -tf.reduce_mean(log_count_probs, name="loss")

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
            self.loss = loss


class EstimatorGraph(TFEstimatorGraph):
    sample_data: tf.Tensor

    mu: tf.Tensor
    sigma2: tf.Tensor
    a: tf.Tensor
    b: tf.Tensor

    def __init__(
            self,
            sample_data,
            design,
            num_samples,
            num_genes,
            num_design_params,
            graph: tf.Graph = None,
            batch_size=500
    ):
        super().__init__(graph)

        # initial graph elements
        with self.graph.as_default():
            # sample_data = tf_ops.caching_placeholder(tf.float32, shape=(num_samples, num_genes), name="sample_data")
            # design = tf_ops.caching_placeholder(tf.float32, shape=(num_samples, num_design_params), name="design")

            learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
            # train_steps = tf.placeholder(tf.int32, shape=(), name="training_steps")

            with tf.name_scope("initialization"):
                # implicit broadcasting of sample_data and initial_mixture_probs to
                #   shape (num_mixtures, num_samples, num_genes)
                init_dist = nb_utils.fit(sample_data, axis=-2)
                assert init_dist.r.shape == [1, num_genes]

            data = tf.data.Dataset.from_tensor_slices((
                tf.range(num_samples, name="sample_index"),
                sample_data,
                design
            ))
            data = data.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=batch_size * 5))
            data = data.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

            iterator = data.make_one_shot_iterator()
            batch_sample_index, batch_sample_data, batch_design = iterator.get_next()

            # Batch model:
            #     only `batch_size` samples will be used;
            #     All per-sample variables have to be passed via `data`.
            #     Sample-independent variables (e.g. per-gene distributions) can be created inside the batch model
            batch_model = LinearBatchModel(
                init_dist,
                batch_sample_data,
                batch_design
            )

            with tf.name_scope("mu_estim"):
                mu_estim = tf.exp(tf.tile(batch_model.a_intercept, (num_samples, 1)))
            with tf.name_scope("r_estim"):
                r_estim = tf.exp(tf.tile(batch_model.b_intercept, (num_samples, 1)))
            dist_estim = nb_utils.NegativeBinomial(mean=mu_estim, r=r_estim)

            with tf.name_scope("mu_obs"):
                mu_obs = tf.exp(tf.matmul(design, batch_model.a))
            with tf.name_scope("r_obs"):
                r_obs = tf.exp(tf.matmul(design, batch_model.b))
            dist_obs = nb_utils.NegativeBinomial(mean=mu_obs, r=r_obs)

            # ### management
            with tf.name_scope('summaries'):
                tf.summary.histogram('a_intercept', batch_model.a_intercept)
                tf.summary.histogram('b_intercept', batch_model.b_intercept)
                tf.summary.histogram('a_slope', batch_model.a_slope)
                tf.summary.histogram('b_slope', batch_model.b_slope)
                tf.summary.scalar('loss', batch_model.loss)

            with tf.name_scope("training"):
                self.global_train_step = tf.train.get_or_create_global_step()
                self.loss = batch_model.loss

                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                self.gradient = self.optimizer.compute_gradients(self.loss)

                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_train_step)

            self.init_ops = []
            self.init_op = tf.variables_initializer(tf.global_variables(), name="init_op")

            self.saver = tf.train.Saver()
            self.merged_summary = tf.summary.merge_all()

            # ### set up class attributes
            self.sample_data = sample_data
            self.design = design

            self.distribution_estim = dist_estim
            self.distribution_obs = dist_obs
            self.batch_model = batch_model

            self.mu = dist_obs.mean()
            self.sigma2 = dist_obs.variance()
            self.a = batch_model.a
            self.b = batch_model.b
            assert (self.mu.shape == (num_samples, num_genes))
            assert (self.sigma2.shape == (num_samples, num_genes))
            assert (self.a.shape == (num_design_params, num_genes))
            assert (self.b.shape == (num_design_params, num_genes))

    def initialize(self, session, feed_dict, **kwargs):
        with self.graph.as_default():
            for op in self.init_ops:
                session.run(op, feed_dict=feed_dict)
            session.run(self.init_op)

    def train(self, session, feed_dict=None, *args, steps=1000, learning_rate=0.05, **kwargs):
        print("learning rate: %s" % learning_rate)

        feed_dict = dict() if feed_dict is None else feed_dict.copy()
        feed_dict["learning_rate:0"] = learning_rate

        loss_res = None
        for i in range(steps):
            (train_step, loss_res, _) = session.run((self.global_train_step, self.loss, self.train_op),
                                                    feed_dict=feed_dict)

            print("Step: %d\tloss: %f" % (train_step, loss_res))

        return loss_res


class Estimator(AbstractEstimator, TFEstimator, metaclass=abc.ABCMeta):
    model: EstimatorGraph

    scaffold: tf.train.Scaffold
    hooks: List[tf.train.SessionRunHook]

    def __init__(self, input_data: Union[xr.Dataset, anndata.AnnData],
                 batch_size=500,
                 model=None,
                 working_dir=None):
        if working_dir is None:
            working_dir = os.path.join("data/log/", self.__module__,
                                       datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
            if not os.path.exists(working_dir):
                os.makedirs(working_dir)
        self.working_dir = working_dir

        if model is None:
            tf.reset_default_graph()

            # read input_data
            if isinstance(input_data, xr.Dataset):
                num_genes = input_data.dims["genes"]
                num_samples = input_data.dims["samples"]
                num_design_params = input_data.dims["design_params"]

                sample_data = np.asarray(input_data["sample_data"], dtype=np.float32)
                design = np.asarray(input_data["design"], dtype=np.float32)
            elif anndata is not None and isinstance(input_data, anndata.AnnData):
                num_genes = input_data.n_vars
                num_samples = input_data.n_obs
                num_design_params = input_data.obsm["design"].shape[-1]

                # TODO: load sample_data as sparse array instead of casting it to a dense numpy array
                sample_data = np.asarray(input_data.X, dtype=np.float32)
                design = np.asarray(input_data.obsm["design"], dtype=np.float32)
            else:
                raise ValueError("input_data has to be an instance of either xarray.Dataset or anndata.AnnData")

            # create model
            model = EstimatorGraph(
                sample_data=sample_data,
                design=design,
                num_samples=num_samples, num_genes=num_genes, num_design_params=num_design_params,
                batch_size=batch_size,
                graph=tf.get_default_graph()
            )

        # set up session parameters
        with model.graph.as_default():
            self.scaffold = tf.train.Scaffold(
                init_op=model.init_op,
                summary_op=model.merged_summary,
                saver=model.saver,
            )
            self.hooks = [
                TimedRunHook(
                    run_steps=11,
                    call_request_tensors={p: model.__getattribute__(p) for p in PARAMS.keys()},
                    call_fn=lambda sess, step, data: self._save_timestep(step, data)
                ),
                tf.train.NanTensorHook(model.loss),
            ]

        TFEstimator.__init__(self, input_data, model)

    def create_new_session(self):
        self.close_session()
        self.feed_dict = {}
        with self.model.graph.as_default():
            self.session = tf.train.MonitoredTrainingSession(
                checkpoint_dir=self.working_dir,
                scaffold=self.scaffold,
                hooks=self.hooks,
                save_checkpoint_steps=10,
                save_summaries_steps=10,
            )

    def _save_timestep(self, step, data: dict):
        tf_utils.save_timestep(step=step, data=data, params=PARAMS, working_dir=self.working_dir)

    def to_xarray(self, params=PARAMS):
        model_params = {p: self.model.__getattribute__(p) for p in params.keys()}

        data = self.session.run(model_params, feed_dict=None)

        output = {key: (dim, data[key]) for (key, dim) in params.items()}
        output = xr.Dataset(output)

        return output

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
