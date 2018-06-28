import abc
from typing import Union, Dict, Tuple
import logging

import xarray as xr
import tensorflow as tf
import numpy as np
import scipy.sparse

try:
    import anndata
except ImportError:
    anndata = None

import utils.stats as stat_utils
from .external import AbstractEstimator, MonitoredTFEstimator, TFEstimatorGraph
from .external import nb_utils, tf_linreg

ESTIMATOR_PARAMS = AbstractEstimator.params().copy()
ESTIMATOR_PARAMS.update({
    "batch_count_probs": ("batch_samples", "genes"),
    "batch_log_count_probs": ("batch_samples", "genes"),
    "batch_log_likelihood": (),
    "full_loss": (),
    "full_gradient": ("genes",),
})

logger = logging.getLogger(__name__)


# session / device config
# CONFIG = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

class BasicModel:

    def __init__(self, sample_data, design, a, b):
        dist_estim = nb_utils.NegativeBinomial(mean=tf.exp(tf.gather(a, 0)),
                                               r=tf.exp(tf.gather(b, 0)),
                                               name="dist_estim")

        with tf.name_scope("mu"):
            log_mu = tf.matmul(design, a, name="log_mu_obs")
            log_mu = tf.clip_by_value(
                log_mu,
                np.log(np.nextafter(0, 1, dtype=log_mu.dtype.as_numpy_dtype)),
                np.log(log_mu.dtype.max)
            )
            mu = tf.exp(log_mu)

        with tf.name_scope("r"):
            log_r = tf.matmul(design, b, name="log_r_obs")
            log_r = tf.clip_by_value(
                log_r,
                np.log(np.nextafter(0, 1, dtype=log_r.dtype.as_numpy_dtype)),
                np.log(log_r.dtype.max)
            )
            r = tf.exp(log_r)

        dist_obs = nb_utils.NegativeBinomial(mean=mu, r=r, name="dist_obs")

        with tf.name_scope("count_probs"):
            count_probs = dist_obs.prob(sample_data)
            count_probs = tf.clip_by_value(
                count_probs,
                0.,
                1.
            )

        with tf.name_scope("log_count_probs"):
            log_count_probs = dist_obs.log_prob(sample_data)
            log_count_probs = tf.clip_by_value(
                log_count_probs,
                np.log(np.nextafter(0, 1, dtype=log_count_probs.dtype.as_numpy_dtype)),
                0.
            )

        self.sample_data = sample_data
        self.design = design

        self.dist_estim = dist_estim
        self.mu_estim = dist_estim.mean()
        self.r_estim = dist_estim.r
        self.sigma2_estim = dist_estim.variance()

        self.dist_obs = dist_obs
        self.mu = mu
        self.r = r
        self.sigma2 = dist_obs.variance()

        self.count_probs = count_probs
        self.log_count_probs = log_count_probs
        self.log_likelihood = tf.reduce_sum(self.log_count_probs, name="log_likelihood")

        with tf.name_scope("loss"):
            self.loss = -tf.reduce_mean(self.log_count_probs)


class ModelVars:
    a: tf.Tensor
    a_intercept: tf.Variable
    a_slope: tf.Variable
    b: tf.Tensor
    b_intercept: tf.Variable
    b_slope: tf.Variable

    def __init__(
            self,
            sample_data,
            design,
            name="Linear_Batch_Model",
            init_a_intercept=None,
            init_a_slopes=None,
            init_b_intercept=None,
            init_b_slopes=None,
    ):
        with tf.name_scope(name):
            num_design_params = design.shape[-1]
            (batch_size, num_genes) = sample_data.shape

            assert sample_data.shape == [batch_size, num_genes]
            assert design.shape == [batch_size, num_design_params]

            with tf.name_scope("initialization"):
                # implicit broadcasting of sample_data and initial_mixture_probs to
                #   shape (num_mixtures, num_samples, num_genes)
                init_dist = nb_utils.fit(sample_data, axis=-2)
                assert init_dist.r.shape == [1, num_genes]

                if init_a_intercept is None:
                    init_a_intercept = tf.log(init_dist.mean())
                    init_a_intercept = tf.clip_by_value(
                        init_a_intercept,
                        clip_value_min=np.log(1),
                        clip_value_max=np.log(init_a_intercept.dtype.max) / 8
                    )
                else:
                    init_a_intercept = tf.convert_to_tensor(init_a_intercept, dtype=sample_data.dtype)

                if init_b_intercept is None:
                    init_b_intercept = tf.log(init_dist.r)
                    init_b_intercept = tf.clip_by_value(
                        init_b_intercept,
                        clip_value_min=np.log(1),
                        clip_value_max=np.log(init_a_intercept.dtype.max) / 8
                    )
                else:
                    init_b_intercept = tf.convert_to_tensor(init_b_intercept, dtype=sample_data.dtype)
                assert init_b_intercept.shape == [1, num_genes] == init_b_intercept.shape

                if init_a_slopes is None:
                    init_a_slopes = tf.random_uniform(
                        tf.TensorShape([num_design_params - 1, num_genes]),
                        minval=np.nextafter(0, 1, dtype=design.dtype.as_numpy_dtype),
                        maxval=np.sqrt(np.nextafter(0, 1, dtype=design.dtype.as_numpy_dtype)),
                        dtype=design.dtype
                    )
                else:
                    init_a_slopes = tf.convert_to_tensor(init_a_slopes, dtype=sample_data.dtype)

                if init_b_slopes is None:
                    init_b_slopes = init_a_slopes
                else:
                    init_b_slopes = tf.convert_to_tensor(init_b_slopes, dtype=sample_data.dtype)

            a, a_intercept, a_slope = tf_linreg.param_variable(init_a_intercept, init_a_slopes, name="a")
            b, b_intercept, b_slope = tf_linreg.param_variable(init_b_intercept, init_b_slopes, name="b")
            assert a.shape == (num_design_params, num_genes) == b.shape

            self.a = a
            self.a_intercept = a_intercept
            self.a_slope = a_slope
            self.b = b
            self.b_intercept = b_intercept
            self.b_slope = b_slope


def fetch_batch(indices, sample_data, design):
    batch_sample_data = tf.gather(sample_data, indices)
    batch_design = tf.gather(design, indices)
    return indices, (batch_sample_data, batch_design)


def map_reduce(last_elem: tf.Tensor, data: tf.data.Dataset, map_fn, reduce_fn=tf.add, **kwargs):
    iterator = data.make_initializable_iterator()

    def cond(idx, val):
        return tf.not_equal(tf.gather(idx, tf.size(idx) - 1), last_elem)

    def body_fn(old_idx, old_val):
        idx, val = iterator.get_next()

        return idx, reduce_fn(old_val, map_fn(idx, val))

    def init_vals():
        idx, val = iterator.get_next()
        return idx, map_fn(idx, val)

    with tf.control_dependencies([iterator.initializer]):
        _, reduced = tf.while_loop(cond, body_fn, init_vals(), **kwargs)

    return reduced


class FullDataModel:
    def __init__(
            self,
            sample_indices: tf.Tensor,
            fetch_fn,
            batch_size: Union[int, tf.Tensor],
            a: tf.Tensor,
            b: tf.Tensor
    ):
        dataset = tf.data.Dataset.from_tensor_slices(sample_indices)

        batched_data = dataset.batch(batch_size)
        batched_data = batched_data.map(fetch_fn)
        batched_data = batched_data.prefetch(1)

        def map_model(idx, data) -> BasicModel:
            data, design = data
            model = BasicModel(data, design, a, b)
            return model

        super()
        model = map_model(*fetch_fn(sample_indices))

        with tf.name_scope("log_likelihood"):
            log_likelihood = map_reduce(
                last_elem=tf.gather(sample_indices, tf.size(sample_indices) - 1),
                data=batched_data,
                map_fn=lambda idx, data: map_model(idx, data).log_likelihood,
                parallel_iterations=1,
            )

        with tf.name_scope("loss"):
            loss = -map_reduce(
                last_elem=tf.gather(sample_indices, tf.size(sample_indices) - 1),
                data=batched_data,
                map_fn=lambda idx, data: map_model(idx, data).log_likelihood,
                parallel_iterations=1,
            )
            loss = loss / tf.cast(tf.size(sample_indices), dtype=loss.dtype)

        self.sample_data = model.sample_data
        self.design = model.design

        self.dist_estim = model.dist_estim
        self.mu_estim = model.mu_estim
        self.r_estim = model.r_estim
        self.sigma2_estim = model.sigma2_estim

        self.dist_obs = model.dist_obs
        self.mu = model.mu
        self.r = model.r
        self.sigma2 = model.sigma2

        self.count_probs = model.count_probs
        self.log_count_probs = model.log_count_probs

        # custom
        self.sample_indices = sample_indices

        self.log_likelihood = log_likelihood
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
            batch_size=500,
            init_a_intercept=None,
            init_a_slopes=None,
            init_b_intercept=None,
            init_b_slopes=None,
            extended_summary=False,
    ):
        super().__init__(graph)
        self.num_samples = num_samples
        self.num_genes = num_genes
        self.num_design_params = num_design_params
        self.batch_size = batch_size

        # initial graph elements
        with self.graph.as_default():
            # design = tf_ops.caching_placeholder(tf.float32, shape=(num_samples, num_design_params), name="design")

            learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
            # train_steps = tf.placeholder(tf.int32, shape=(), name="training_steps")

            with tf.name_scope("input_pipeline"):
                data_indices = tf.data.Dataset.from_tensor_slices((
                    tf.range(num_samples, name="sample_index")
                ))
                training_data = data_indices.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=2 * batch_size))
                training_data = training_data.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
                training_data = training_data.map(lambda indices: fetch_batch(indices, sample_data, design))
                training_data = training_data.prefetch(2)

                iterator = training_data.make_one_shot_iterator()
                batch_sample_index, (batch_sample_data, batch_design) = iterator.get_next()

            # Batch model:
            #     only `batch_size` samples will be used;
            #     All per-sample variables have to be passed via `data`.
            #     Sample-independent variables (e.g. per-gene distributions) can be created inside the batch model
            batch_vars = ModelVars(
                batch_sample_data,
                batch_design,
                init_a_intercept=init_a_intercept,
                init_a_slopes=init_a_slopes,
                init_b_intercept=init_b_intercept,
                init_b_slopes=init_b_slopes,
            )

            batch_model = BasicModel(batch_sample_data, batch_design, batch_vars.a, batch_vars.b)

            # minimize negative log probability (log(1) = 0);
            # use the mean loss to keep a constant learning rate independently of the batch size
            loss = -tf.reduce_mean(batch_model.log_count_probs, name="loss")

            # ### management
            with tf.name_scope("training"):
                global_step = tf.train.get_or_create_global_step()

                gradient = tf.gradients(loss, tf.trainable_variables())
                gradient = [(g, v) for g, v in zip(gradient, tf.trainable_variables())]

                aggregated_gradient = tf.add_n([tf.reduce_sum(tf.abs(grad), axis=0) for (grad, var) in gradient])
                # max_gradient = tf.reduce_max(
                #     tf.concat([tf.reduce_max(tf.abs(grad)) for (grad, var) in gradient], axis=0)
                # )

                # # smooth loss
                # loss_ema = tf.train.ExponentialMovingAverage(decay=0.7, zero_debias=True)
                # apply_loss_ema = loss_ema.apply([loss])
                # smoothed_loss = loss_ema.average(loss)
                #
                # with tf.control_dependencies([apply_loss_ema]):
                optim_GD = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
                optim_Adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
                optim_Adagrad = tf.train.AdagradOptimizer(learning_rate=learning_rate)
                optim_RMSProp = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

                train_op_GD = optim_GD.apply_gradients(gradient, global_step=global_step)
                train_op_Adam = optim_Adam.apply_gradients(gradient, global_step=global_step)
                train_op_Adagrad = optim_Adagrad.apply_gradients(gradient, global_step=global_step)
                train_op_RMSProp = optim_RMSProp.apply_gradients(gradient, global_step=global_step)

            with tf.name_scope("init_op"):
                init_op = tf.global_variables_initializer()
                # with tf.control_dependencies([init_op]):
                #     init_op = tf.assign(smoothed_loss, loss)

            self.batch_vars = batch_vars
            self.batch_model = batch_model

            self.loss = loss
            # self.smoothed_loss = smoothed_loss

            self.global_step = global_step

            self.gradient = aggregated_gradient
            self.plain_gradient = gradient

            self.train_op_GD = train_op_GD
            self.train_op_Adam = train_op_Adam
            self.train_op_Adagrad = train_op_Adagrad
            self.train_op_RMSProp = train_op_RMSProp
            # default train op
            self.train_op = train_op_GD

            self.init_ops = []
            self.init_op = init_op

            # ### set up class attributes
            self.sample_data = sample_data
            self.design = design

            # self.distribution_estim = batch_model.dist_estim
            # self.distribution_obs = batch_model.dist_obs

            # self.mu = batch_model.dist_obs.mean()
            # self.r = batch_model.dist_obs.r
            # self.sigma2 = batch_model.dist_obs.variance()
            self.a = batch_vars.a
            self.b = batch_vars.b
            # assert (self.mu.shape == (num_samples, num_genes))
            # assert (self.sigma2.shape == (num_samples, num_genes))
            assert (self.a.shape == (num_design_params, num_genes))
            assert (self.b.shape == (num_design_params, num_genes))

            self.batch_count_probs = batch_model.count_probs
            self.batch_log_count_probs = batch_model.log_count_probs
            self.batch_log_likelihood = batch_model.log_likelihood

            # ### alternative definitions for custom samples:
            sample_selection = tf.placeholder_with_default(tf.range(num_samples),
                                                           shape=(None,),
                                                           name="sample_selection")
            full_data = FullDataModel(
                sample_indices=sample_selection,
                fetch_fn=lambda indices: fetch_batch(indices, sample_data, design),
                batch_size=batch_size,
                a=self.a,
                b=self.b,
            )
            full_gradient_a, full_gradient_b = tf.gradients(full_data.loss, (batch_vars.a, batch_vars.b))
            full_gradient = (
                    tf.reduce_sum(tf.abs(full_gradient_a), axis=0) +
                    tf.reduce_sum(tf.abs(full_gradient_b), axis=0)
            )

            self.sample_selection = sample_selection
            self.full_data = full_data

            self.mu = full_data.mu
            self.r = full_data.r
            self.sigma2 = full_data.sigma2

            self.full_gradient = full_gradient
            self.full_loss = full_data.loss

            with tf.name_scope('summaries'):
                tf.summary.histogram('a_intercept', batch_vars.a_intercept)
                tf.summary.histogram('b_intercept', batch_vars.b_intercept)
                tf.summary.histogram('a_slope', batch_vars.a_slope)
                tf.summary.histogram('b_slope', batch_vars.b_slope)
                tf.summary.scalar('loss', loss)
                tf.summary.scalar('learning_rate', learning_rate)

                if extended_summary:
                    tf.summary.scalar('median_ll',
                                      tf.contrib.distributions.percentile(
                                          tf.reduce_sum(batch_model.log_count_probs, axis=1),
                                          50.)
                                      )
                    tf.summary.histogram('gradient_a', tf.gradients(loss, batch_vars.a))
                    tf.summary.histogram('gradient_b', tf.gradients(loss, batch_vars.b))
                    tf.summary.histogram("full_gradient", full_gradient)
                    tf.summary.scalar("full_gradient_median", tf.contrib.distributions.percentile(full_gradient, 50.))
                    tf.summary.scalar("full_gradient_mean", tf.reduce_mean(full_gradient))

            self.saver = tf.train.Saver()
            self.merged_summary = tf.summary.merge_all()


class Estimator(AbstractEstimator, MonitoredTFEstimator, metaclass=abc.ABCMeta):
    model: EstimatorGraph

    @classmethod
    def params(cls) -> dict:
        return ESTIMATOR_PARAMS

    def __init__(self, input_data: Union[xr.Dataset, anndata.AnnData, Dict[Tuple[int, int], any]],
                 batch_size=500,
                 design_matrix=None,
                 design_key="design",
                 model=None,
                 graph=None,
                 init_a_intercept=None,
                 init_a_slopes=None,
                 init_b_intercept=None,
                 init_b_slopes=None,
                 extended_summary=False,
                 ):

        if model is None:
            if graph is None:
                graph = tf.Graph()

            # read input_data
            if anndata is not None and isinstance(input_data, anndata.AnnData):
                sample_data = input_data.X
                if design_matrix is None:
                    design_matrix = input_data.obsm[design_key]

                num_genes = input_data.n_vars
                num_samples = input_data.n_obs
                num_design_params = design_matrix.shape[-1]
            elif isinstance(input_data, xr.Dataset):
                sample_data = np.asarray(input_data["sample_data"].values, dtype=np.float32)
                if design_matrix is None:
                    design_matrix = np.asarray(input_data[design_key], dtype=np.float32)

                num_genes = input_data.dims["genes"]
                num_samples = input_data.dims["samples"]
                num_design_params = input_data.dims["design_params"]
            else:
                sample_data = np.asarray(input_data, dtype=np.float32)

                (num_samples, num_genes) = input_data.shape
                num_design_params = design_matrix.shape[-1]

            design_matrix = np.asarray(design_matrix, dtype=np.float32)
            self._sample_data = sample_data
            self._design = design_matrix

            with graph.as_default():
                if scipy.sparse.issparse(sample_data):
                    # coo = sample_data.tocoo()
                    # indices = np.mat([coo.row, coo.col]).transpose()
                    # sample_data = tf.SparseTensor(indices, coo.data, coo.shape)

                    # ### Convert to dense matrix to reduce overhead.
                    sample_data = tf.convert_to_tensor(sample_data.toarray(), dtype=tf.float32)
                else:
                    sample_data = tf.convert_to_tensor(sample_data, dtype=tf.float32)

                # create model
                model = EstimatorGraph(
                    sample_data=sample_data,
                    design=design_matrix,
                    num_samples=num_samples, num_genes=num_genes, num_design_params=num_design_params,
                    batch_size=batch_size,
                    graph=graph,
                    init_a_intercept=init_a_intercept,
                    init_a_slopes=init_a_slopes,
                    init_b_intercept=init_b_intercept,
                    init_b_slopes=init_b_slopes,
                    extended_summary=extended_summary
                )

        MonitoredTFEstimator.__init__(self, input_data, model)

    def _scaffold(self):
        with self.model.graph.as_default():
            scaffold = tf.train.Scaffold(
                init_op=self.model.init_op,
                summary_op=self.model.merged_summary,
                saver=self.model.saver,
            )
        return scaffold

    def train(self, *args,
              learning_rate=0.5,
              convergence_criteria="t_test",
              loss_history_size=200,
              stop_at_loss_change=0.05,
              **kwargs):
        super().train(*args,
                      feed_dict={"learning_rate:0": learning_rate},
                      convergence_criteria=convergence_criteria,
                      loss_history_size=loss_history_size,
                      stop_at_loss_change=stop_at_loss_change,
                      **kwargs)

    @property
    def sample_data(self):
        return self._sample_data

    @property
    def design(self):
        return self._design

    @property
    def mu(self):
        return self.get("mu")

    @property
    def r(self):
        return self.get("r")

    @property
    def sigma2(self):
        return self.get("sigma2")

    @property
    def a(self):
        return self.get("a")

    @property
    def b(self):
        return self.get("b")

    @property
    def gradient(self):
        return self._get_unsafe("gradient")

    def count_probs(self, sample_data=None, sample_indices=None):
        if sample_data is None:
            if sample_indices is None:
                sample_indices = np.arange(self.model.num_samples)
            feed_dict = {self.model.sample_selection: sample_indices}

            return self.run(self.model.full_data.count_probs, feed_dict=feed_dict)
        else:
            return super().count_probs(sample_data)

    def log_count_probs(self, sample_data=None, sample_indices=None):
        if sample_data is None:
            if sample_indices is None:
                sample_indices = np.arange(self.model.num_samples)
            feed_dict = {self.model.sample_selection: sample_indices}

            return self.run(self.model.full_data.log_count_probs, feed_dict=feed_dict)
        else:
            return super().log_count_probs(sample_data)

    def log_likelihood(self, sample_data=None, sample_indices=None):
        if sample_data is None:
            if sample_indices is None:
                sample_indices = np.arange(self.model.num_samples)
            feed_dict = {self.model.sample_selection: sample_indices}

            return self.run(self.model.full_data.log_likelihood, feed_dict=feed_dict)
        else:
            return super().log_likelihood(sample_data)

    @property
    def full_loss(self):
        return self._get_unsafe("full_loss")

    @property
    def full_gradient(self):
        return self._get_unsafe("full_gradient")
