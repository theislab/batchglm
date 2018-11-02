import abc
from typing import Union, Dict, Tuple, List
import logging
import pprint
from enum import Enum

import tensorflow as tf
# import tensorflow_probability as tfp

import numpy as np
import xarray as xr

try:
    import anndata
except ImportError:
    anndata = None

from .base import BasicModelGraph, ModelVars, MixtureModel, ESTIMATOR_PARAMS
from .base import param_bounds, tf_clip_param, np_clip_param

from .external import AbstractEstimator, XArrayEstimatorStore, InputData, Model, MonitoredTFEstimator, TFEstimatorGraph
from .external import nb_utils, train_utils, op_utils, rand_utils, linalg_utils, data_utils, nb_glm_utils
from .external import pkg_constants

logger = logging.getLogger(__name__)


class FullDataModelGraph:
    def __init__(
            self,
            sample_indices: tf.Tensor,
            fetch_fn,
            batch_size: Union[int, tf.Tensor],
            model_vars: ModelVars,
    ):
        dataset = tf.data.Dataset.from_tensor_slices(sample_indices)

        batched_data = dataset.batch(batch_size)
        batched_data = batched_data.map(fetch_fn, num_parallel_calls=pkg_constants.TF_NUM_THREADS)
        batched_data = batched_data.prefetch(1)

        def map_model(idx, data) -> BasicModelGraph:
            X, design_loc, design_scale, size_factors = data
            model = BasicModelGraph(
                X=X,
                design_loc=design_loc,
                design_scale=design_scale,
                design_mixture_loc=model_vars.design_mixture_loc,
                design_mixture_scale=model_vars.design_mixture_scale,
                a=model_vars.a,
                b=model_vars.b,
                mixture_logits=model_vars.mixture_logits,
                size_factors=size_factors
            )
            return model

        model = map_model(*fetch_fn(sample_indices))

        with tf.name_scope("log_likelihood"):
            log_likelihood = op_utils.map_reduce(
                last_elem=tf.gather(sample_indices, tf.size(sample_indices) - 1),
                data=batched_data,
                map_fn=lambda idx, data: map_model(idx, data).log_likelihood,
                parallel_iterations=1,
            )
            norm_log_likelihood = log_likelihood / tf.cast(tf.size(sample_indices), dtype=log_likelihood.dtype)
            norm_neg_log_likelihood = - norm_log_likelihood

        with tf.name_scope("loss"):
            loss = tf.reduce_sum(norm_neg_log_likelihood)

        self.X = model.X
        self.design_loc = model.design_loc
        self.design_scale = model.design_scale

        self.batched_data = batched_data

        self.dist_estim = model.dist_estim
        self.mu_estim = model.mu_estim
        self.r_estim = model.r_estim
        self.sigma2_estim = model.sigma2_estim

        self.dist_obs = model.dist_obs
        self.mu = model.mu
        self.r = model.r
        self.sigma2 = model.sigma2

        self.probs = tf.exp(model.log_probs)
        self.log_probs = model.log_probs

        # custom
        self.sample_indices = sample_indices

        self.log_likelihood = log_likelihood
        self.norm_log_likelihood = norm_log_likelihood
        self.norm_neg_log_likelihood = norm_neg_log_likelihood
        self.loss = loss


class EstimatorGraph(TFEstimatorGraph):
    sample_data: tf.Tensor
    design: tf.Tensor

    mu: tf.Tensor
    sigma2: tf.Tensor
    a: tf.Tensor
    b: tf.Tensor
    mixture_prob: tf.Tensor
    mixture_assignment: tf.Tensor

    def __init__(
            self,
            fetch_fn,
            feature_isnonzero,
            num_mixtures,
            num_observations,
            num_features,
            num_design_loc_params,
            num_design_scale_params,
            num_design_mixture_loc_params,
            num_design_mixture_scale_params,
            design_mixture_loc,
            design_mixture_scale,
            graph: tf.Graph = None,
            batch_size=500,
            init_a=None,
            init_b=None,
            init_mixture_weights=None,
            extended_summary=False,
            dtype="float32"
    ):
        super().__init__(graph)

        self.num_mixtures = num_mixtures
        self.num_observations = num_observations
        self.num_features = num_features
        self.num_design_loc_params = num_design_loc_params
        self.num_design_scale_params = num_design_scale_params
        self.num_design_mixture_loc_params = num_design_mixture_loc_params
        self.num_design_mixture_scale_params = num_design_mixture_scale_params
        self.batch_size = batch_size

        # initial graph elements
        with self.graph.as_default():
            # ### placeholders
            learning_rate = tf.placeholder(dtype, shape=(), name="learning_rate")

            # ### performance related settings
            buffer_size = 4

            init_dist = nb_utils.NegativeBinomial(
                mean=tf.random_uniform(
                    minval=10,
                    maxval=1000,
                    shape=[1, num_features],
                    dtype=dtype
                ),
                r=tf.random_uniform(
                    minval=1,
                    maxval=10,
                    shape=[1, num_features],
                    dtype=dtype
                ),
            )
            assert init_dist.r.shape == [1, num_features]

            model_vars = ModelVars(
                init_dist=init_dist,
                dtype=dtype,
                num_mixtures=num_mixtures,
                num_observations=num_observations,
                num_features=num_features,
                num_design_loc_params=num_design_loc_params,
                num_design_scale_params=num_design_scale_params,
                num_design_mixture_loc_params=num_design_mixture_loc_params,
                num_design_mixture_scale_params=num_design_mixture_scale_params,
                design_mixture_loc=design_mixture_loc,
                design_mixture_scale=design_mixture_scale,
                init_mixture_weights=init_mixture_weights,
                init_a=init_a,
                init_b=init_b,
            )

            with tf.name_scope("input_pipeline"):
                data_indices = tf.data.Dataset.from_tensor_slices((
                    tf.range(num_observations, name="sample_index")
                ))
                training_data = data_indices.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=2 * batch_size))
                # training_data = training_data.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
                training_data = training_data.batch(batch_size, drop_remainder=True)
                training_data = training_data.map(fetch_fn, num_parallel_calls=pkg_constants.TF_NUM_THREADS)
                training_data = training_data.prefetch(buffer_size)

                iterator = training_data.make_one_shot_iterator()

                batch_sample_index, batch_data = iterator.get_next()
                (batch_X, batch_design_loc, batch_design_scale, batch_size_factors) = batch_data
                batch_mixture_logits = tf.gather(
                    model_vars.mixture_logits,
                    indices=batch_sample_index,
                    axis=0,
                    name="batch_mixture_logits"
                )

            with tf.name_scope("batch"):
                # Batch model:
                #   only `batch_size` observations will be used;
                #   All per-sample variables have to be passed via `data`.
                #   Sample-independent variables (e.g. per-feature distributions) can be created inside the batch model
                batch_model = BasicModelGraph(
                    X=batch_X,
                    design_loc=batch_design_loc,
                    design_scale=batch_design_scale,
                    design_mixture_loc=model_vars.design_mixture_loc,
                    design_mixture_scale=model_vars.design_mixture_scale,
                    a=model_vars.a,
                    b=model_vars.b,
                    mixture_logits=batch_mixture_logits,
                    size_factors=batch_size_factors
                )

                # minimize negative log probability (log(1) = 0);
                # use the mean loss to keep a constant learning rate independently of the batch size
                batch_loss = batch_model.loss

            with tf.name_scope("full_data"):
                # ### alternative definitions for custom observations:
                sample_selection = tf.placeholder_with_default(tf.range(num_observations),
                                                               shape=(None,),
                                                               name="sample_selection")
                full_data_model = FullDataModelGraph(
                    sample_indices=sample_selection,
                    fetch_fn=fetch_fn,
                    batch_size=batch_size * buffer_size,
                    model_vars=model_vars,
                )
                full_data_loss = full_data_model.loss

                mu = full_data_model.mu
                r = full_data_model.r
                sigma2 = full_data_model.sigma2
                mixture_model = MixtureModel(model_vars.mixture_logits)

            # ### management
            with tf.name_scope("training"):
                global_step = tf.train.get_or_create_global_step()

                # set up trainers for different selections of variables to train
                # set up multiple optimization algorithms for each trainer
                batch_trainers = train_utils.MultiTrainer(
                    loss=batch_model.norm_neg_log_likelihood,
                    variables=[model_vars.a_var, model_vars.b_var, model_vars.mixture_logits_var],
                    learning_rate=learning_rate,
                    global_step=global_step,
                    name="batch_trainers"
                )
                with tf.name_scope("batch_gradient"):
                    # use same gradient as the optimizers
                    batch_gradient_a = batch_trainers.gradient[0][0]
                    batch_gradient_a = tf.reduce_sum(tf.abs(batch_gradient_a), axis=[0, 1])
                    batch_gradient_b = batch_trainers.gradient[1][0]
                    batch_gradient_b = tf.reduce_sum(tf.abs(batch_gradient_b), axis=[0, 1])
                    batch_gradient = tf.add(batch_gradient_a, batch_gradient_b)

                full_data_trainers = train_utils.MultiTrainer(
                    loss=full_data_model.norm_neg_log_likelihood,
                    variables=[model_vars.a_var, model_vars.b_var, model_vars.mixture_logits_var],
                    learning_rate=learning_rate,
                    global_step=global_step,
                    name="full_data_trainers"
                )
                with tf.name_scope("full_gradient"):
                    # use same gradient as the optimizers
                    full_gradient_a = full_data_trainers.gradient[0][0]
                    full_gradient_a = tf.reduce_sum(tf.abs(full_gradient_a), axis=[0, 1])
                    full_gradient_b = full_data_trainers.gradient[1][0]
                    full_gradient_b = tf.reduce_sum(tf.abs(full_gradient_b), axis=[0, 1])
                    full_gradient = tf.add(full_gradient_a, full_gradient_b)

            with tf.name_scope("training"):
                self.global_train_step = tf.train.get_or_create_global_step()
                self.loss = batch_model.loss

                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                self.gradient = self.optimizer.compute_gradients(self.loss)

                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_train_step)

            with tf.name_scope("init_op"):
                init_op = tf.global_variables_initializer()

            # # ### output values:
            # #       override all-zero features with lower bound coefficients
            # with tf.name_scope("output"):
            #     bounds_min, bounds_max = param_bounds(dtype)
            #
            #     param_nonzero_a = tf.broadcast_to(
            #         feature_isnonzero,
            #         shape=[num_mixtures, num_design_loc_params, num_features]
            #     )
            #     alt_a = tf.concat([
            #         # intercept
            #         tf.broadcast_to(bounds_min["a"], [num_mixtures, 1, num_features]),
            #         # slope
            #         tf.zeros(shape=[num_mixtures, num_design_loc_params - 1, num_features], dtype=model_vars.a.dtype),
            #     ], axis=-2, name="alt_a")
            #     a = tf.where(
            #         param_nonzero_a,
            #         model_vars.a,
            #         alt_a,
            #         name="a"
            #     )
            #
            #     param_nonzero_b = tf.broadcast_to(
            #         feature_isnonzero,
            #         shape=[num_mixtures, num_design_scale_params, num_features]
            #     )
            #     alt_b = tf.concat([
            #         # intercept
            #         tf.broadcast_to(bounds_max["b"], [num_mixtures, 1, num_features]),
            #         # slope
            #         tf.zeros(shape=[num_mixtures, num_design_scale_params - 1, num_features], dtype=model_vars.b.dtype),
            #     ], axis=-2, name="alt_b")
            #     b = tf.where(
            #         param_nonzero_b,
            #         model_vars.b,
            #         alt_b,
            #         name="b"
            #     )
            a = model_vars.a
            b = model_vars.b

            self.fetch_fn = fetch_fn
            self.model_vars = model_vars
            self.batch_model = batch_model

            self.learning_rate = learning_rate
            self.loss = batch_loss

            self.batch_trainers = batch_trainers
            self.full_data_trainers = full_data_trainers
            self.global_step = global_step

            self.gradient = batch_gradient
            # self.gradient_a = batch_gradient_a
            # self.gradient_b = batch_gradient_b

            self.train_op = batch_trainers.train_op_GD

            self.init_ops = []
            self.init_op = init_op

            # # ### set up class attributes
            self.a = a
            self.b = b
            self.mixture_logits = model_vars.mixture_logits
            assert (self.a.shape == (num_design_loc_params, num_design_mixture_loc_params, num_features))
            assert (self.b.shape == (num_design_scale_params, num_design_mixture_scale_params, num_features))
            assert (self.mixture_logits.shape == (num_observations, num_mixtures))

            self.mu = mu
            self.r = r
            self.sigma2 = sigma2

            self.mixture_prob = mixture_model.prob
            self.mixture_log_prob = mixture_model.log_prob
            self.mixture_logit_prob = mixture_model.logit_prob
            self.mixture_assignment = mixture_model.mixture_assignment

            self.batch_probs = batch_model.probs
            self.batch_log_probs = batch_model.log_probs
            self.batch_log_likelihood = batch_model.norm_log_likelihood

            self.sample_selection = sample_selection
            self.full_data_model = full_data_model

            self.full_loss = full_data_loss

            self.full_gradient = full_gradient
            # self.full_gradient_a = full_gradient_a
            # self.full_gradient_b = full_gradient_b

            self.saver = tf.train.Saver()
            self.merged_summary = tf.summary.merge_all()
            # self.summary_writer = tf.summary.FileWriter(log_dir, self.graph)

            with tf.name_scope('summaries'):
                tf.summary.histogram('a_intercept', model_vars.a[:, 0])
                tf.summary.histogram('b_intercept', model_vars.b[:, 0])
                tf.summary.histogram('a_slope', model_vars.a[:, 1:])
                tf.summary.histogram('b_slope', model_vars.b[:, 1:])
                tf.summary.scalar('loss', batch_loss)
                tf.summary.scalar('learning_rate', learning_rate)

                with tf.name_scope("prob_image"):
                    # repeat:
                    prob_image = tf.reshape(
                        tf.transpose(tf.tile(
                            [mixture_model.prob],  # input tensor
                            ((num_observations // num_mixtures), 1, 1))),  # target shape
                        [-1]  # flatten
                    )
                    prob_image = tf.transpose(
                        tf.reshape(prob_image, ((num_observations // num_mixtures) * num_mixtures, num_observations)))
                    prob_image = tf.expand_dims(prob_image, 0)
                    prob_image = tf.expand_dims(prob_image, -1)
                    prob_image = prob_image * 255.0

                tf.summary.image('mixture_prob', prob_image)

                if extended_summary:
                    tf.summary.scalar('median_ll',
                                      tf.contrib.distributions.percentile(
                                          tf.reduce_sum(batch_model.log_probs, axis=1),
                                          50.)
                                      )
                    tf.summary.histogram('gradient_a', tf.gradients(batch_loss, model_vars.a))
                    tf.summary.histogram('gradient_b', tf.gradients(batch_loss, model_vars.b))
                    tf.summary.histogram("full_gradient", full_gradient)
                    tf.summary.scalar("full_gradient_median",
                                      tf.contrib.distributions.percentile(full_gradient, 50.))
                    tf.summary.scalar("full_gradient_mean", tf.reduce_mean(full_gradient))
                    tf.summary.scalar('full_loss', full_data_loss)

            self.saver = tf.train.Saver()
            self.merged_summary = tf.summary.merge_all()


class Estimator(AbstractEstimator, MonitoredTFEstimator, metaclass=abc.ABCMeta):
    """
    Estimator for Responding Subset Analysis (RSA)
    """

    class TrainingStrategy(Enum):
        AUTO = None
        DEFAULT = [
            {
                "learning_rate": 0.1,
                "convergence_criteria": "t_test",
                "stopping_criteria": 0.05,
                "loss_window_size": 100,
                "use_batching": True,
                "optim_algo": "ADAM",
            },
            {
                "learning_rate": 0.05,
                "convergence_criteria": "t_test",
                "stopping_criteria": 0.05,
                "loss_window_size": 10,
                "use_batching": False,
                "optim_algo": "ADAM",
            },
        ]
        EXACT = [
            {
                "learning_rate": 0.1,
                "convergence_criteria": "t_test",
                "stopping_criteria": 0.05,
                "loss_window_size": 100,
                "use_batching": True,
                "optim_algo": "ADAM",
            },
            {
                "learning_rate": 0.05,
                "convergence_criteria": "t_test",
                "stopping_criteria": 0.05,
                "loss_window_size": 100,
                "use_batching": True,
                "optim_algo": "ADAM",
            },
            {
                "learning_rate": 0.005,
                "convergence_criteria": "t_test",
                "stopping_criteria": 0.25,
                "loss_window_size": 10,
                "use_batching": False,
                "optim_algo": "Newton-Raphson",
            },
        ]
        QUICK = [
            {
                "learning_rate": 0.1,
                "convergence_criteria": "t_test",
                "stopping_criteria": 0.05,
                "loss_window_size": 100,
                "use_batching": True,
                "optim_algo": "ADAM",
            },
        ]
        PRE_INITIALIZED = [
            {
                "learning_rate": 0.01,
                "convergence_criteria": "t_test",
                "stopping_criteria": 0.25,
                "loss_window_size": 10,
                "use_batching": False,
                "optim_algo": "ADAM",
            },
        ]

    model: EstimatorGraph

    @classmethod
    def param_shapes(cls) -> dict:
        return ESTIMATOR_PARAMS

    def __init__(
            self,
            input_data: InputData,
            batch_size=500,
            graph: tf.Graph = None,
            init_a: Union[np.ndarray, str] = "AUTO",
            init_b: Union[np.ndarray, str] = "AUTO",
            init_mixture_weights: Union[np.ndarray, str] = "AUTO",
            model: EstimatorGraph = None,
            extended_summary=False,
            dtype="float64",
    ):
        if np.linalg.matrix_rank(input_data.design_loc) != np.linalg.matrix_rank(input_data.design_loc.T):
            raise ValueError("design_loc matrix is not full rank")
        if np.linalg.matrix_rank(input_data.design_scale) != np.linalg.matrix_rank(input_data.design_scale.T):
            raise ValueError("design_scale matrix is not full rank")

        if model is None:
            if graph is None:
                graph = tf.Graph()

            self._input_data = input_data

            size_factors_init = input_data.size_factors
            if size_factors_init is not None:
                size_factors_init = np.expand_dims(size_factors_init, axis=1)
                size_factors_init = np.broadcast_to(
                    array=size_factors_init,
                    shape=[input_data.num_observations, input_data.num_features]
                )

            groupwise_means = None  # [groups, features]
            par_link_loc = None
            overall_means = None  # [1, features]
            logger.debug(" * Initialize mean model")
            if isinstance(init_a, str):
                # Chose option if auto was chosen
                if init_a.lower() == "auto":
                    init_a = "closed_form"

                if init_a.lower() == "closed_form":
                    try:
                        groupwise_means, par_link_loc, rmsd_loc = nb_glm_utils.closedform_nb_glm_logmu(
                            X=input_data.X,
                            design_loc=input_data.design_loc,
                            constraints=input_data.constraints_loc,
                            size_factors=size_factors_init
                        )

                        init_a = linalg_utils.stacked_lstsq(
                            L=input_data.design_mixture_loc,
                            b=np.expand_dims(par_link_loc, axis=1)
                        )

                        logger.info("Using closed-form MLE initialization for mean")
                        logger.debug("RMSE of closed-form mean:\n%s", rmsd_loc)
                    except np.linalg.LinAlgError:
                        logger.warning("Closed form initialization failed!")
                elif init_a.lower() == "standard":
                    overall_means = input_data.X.mean(dim="observations").values  # directly calculate the mean
                    # clipping
                    overall_means = np_clip_param(overall_means, "mu")
                    # mean = np.nextafter(0, 1, out=mean, where=mean == 0, dtype=mean.dtype)

                    init_a = np.zeros([
                        input_data.num_design_scale_params,
                        input_data.num_design_mixture_scale_params,
                        input_data.num_features
                    ])
                    init_a[:, 0, :] = np.broadcast_to(
                        np.expand_dims(np.log(overall_means), axis=0),
                        [input_data.num_mixtures, 1, input_data.num_features]
                    )

                    logger.info("Using standard initialization for mean")

            logger.debug(" * Initialize dispersion model")
            if isinstance(init_b, str):
                if init_b.lower() == "auto":
                    init_b = "closed_form"

                if init_b.lower() == "closed_form":
                    try:
                        if par_link_loc is not None:
                            init_a_xr = data_utils.xarray_from_data(
                                par_link_loc,
                                dims=("design_loc_params", "features")
                            )
                            init_a_xr.coords["design_loc_params"] = input_data.design_loc.coords["design_loc_params"]
                            init_mu = np.exp(input_data.design_loc.dot(init_a_xr))
                        else:
                            init_mu = None

                        groupwise_scales, par_link_scale, rmsd_scale = nb_glm_utils.closedform_nb_glm_logphi(
                            X=input_data.X,
                            mu=init_mu,
                            design_scale=input_data.design_scale,
                            constraints=input_data.constraints_loc,
                            size_factors=size_factors_init,
                            groupwise_means=groupwise_means
                        )

                        init_b = linalg_utils.stacked_lstsq(
                            input_data.design_mixture_scale,
                            np.expand_dims(par_link_scale, axis=1)
                        )

                        logger.info("Using closed-form MME initialization for dispersion")
                        logger.debug("RMSE of closed-form dispersion:\n%s", rmsd_scale)
                    except np.linalg.LinAlgError:
                        logger.warning("Closed form initialization failed!")
                elif init_b.lower() == "standard":
                    init_b = np.zeros([
                        input_data.num_design_scale_params,
                        input_data.num_design_mixture_scale_params,
                        input_data.num_features
                    ])

                    logger.info("Using standard initialization for dispersion")

            # ### prepare fetch_fn:
            def fetch_fn(idx):
                r"""
                Documentation of tensorflow coding style in this function:
                tf.py_func defines a python function (the getters of the InputData object slots)
                as a tensorflow operation. Here, the shape of the tensor is lost and
                has to be set with set_shape. For size factors, we use explicit broadcasting
                as explained below.
                """
                # Catch dimension collapse error if idx is only one element long, ie. 0D:
                if len(idx.shape) == 0:
                    idx = tf.expand_dims(idx, axis=0)

                X_tensor = tf.py_func(
                    func=input_data.fetch_X,
                    inp=[idx],
                    Tout=input_data.X.dtype,
                    stateful=False
                )
                X_tensor.set_shape(idx.get_shape().as_list() + [input_data.num_features])
                X_tensor = tf.cast(X_tensor, dtype=dtype)

                design_loc_tensor = tf.py_func(
                    func=input_data.fetch_design_loc,
                    inp=[idx],
                    Tout=input_data.design_loc.dtype,
                    stateful=False
                )
                design_loc_tensor.set_shape(idx.get_shape().as_list() + [input_data.num_design_loc_params])
                design_loc_tensor = tf.cast(design_loc_tensor, dtype=dtype)

                design_scale_tensor = tf.py_func(
                    func=input_data.fetch_design_scale,
                    inp=[idx],
                    Tout=input_data.design_scale.dtype,
                    stateful=False
                )
                design_scale_tensor.set_shape(idx.get_shape().as_list() + [input_data.num_design_scale_params])
                design_scale_tensor = tf.cast(design_scale_tensor, dtype=dtype)

                if input_data.size_factors is not None:
                    size_factors_tensor = tf.log(tf.py_func(
                        func=input_data.fetch_size_factors,
                        inp=[idx],
                        Tout=input_data.size_factors.dtype,
                        stateful=False
                    ))
                    size_factors_tensor.set_shape(idx.get_shape())
                    # Here, we broadcast the size_factor tensor to the batch size,
                    # note that this should not consum any more memory than
                    # keeping the 1D array and performing implicit broadcasting in
                    # the arithmetic operations in the graph.
                    size_factors_tensor = tf.expand_dims(size_factors_tensor, axis=-1)
                    size_factors_tensor = tf.cast(size_factors_tensor, dtype=dtype)
                else:
                    size_factors_tensor = tf.constant(0, shape=[1, 1], dtype=dtype)
                size_factors_tensor = tf.broadcast_to(size_factors_tensor,
                                                      shape=[tf.size(idx), input_data.num_features])

                # return idx, data
                return idx, (X_tensor, design_loc_tensor, design_scale_tensor, size_factors_tensor)

            if isinstance(init_a, str):
                init_a = None
            else:
                init_a = init_a.astype(dtype)
            if isinstance(init_b, str):
                init_b = None
            else:
                init_b = init_b.astype(dtype)
            if isinstance(init_mixture_weights, str):
                init_mixture_weights = None

            logger.debug(" * Start creating model")
            with graph.as_default():
                # create model
                model = EstimatorGraph(
                    fetch_fn=fetch_fn,
                    feature_isnonzero=input_data.feature_isnonzero,
                    num_mixtures=input_data.num_mixtures,
                    num_observations=input_data.num_observations,
                    num_features=input_data.num_features,
                    num_design_loc_params=input_data.num_design_loc_params,
                    num_design_scale_params=input_data.num_design_scale_params,
                    num_design_mixture_loc_params=input_data.num_design_mixture_loc_params,
                    num_design_mixture_scale_params=input_data.num_design_mixture_scale_params,
                    design_mixture_loc=input_data.design_mixture_loc,
                    design_mixture_scale=input_data.design_mixture_scale,
                    graph=graph,
                    batch_size=batch_size,
                    init_a=init_a,
                    init_b=init_b,
                    init_mixture_weights=init_mixture_weights,
                    extended_summary=extended_summary,
                    dtype=dtype
                )
            logger.debug(" * Finished creating model")

            MonitoredTFEstimator.__init__(self, model)

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
              loss_window_size=100,
              stopping_criteria=0.05,
              use_batching=True,
              optim_algo="gradient_descent",
              **kwargs):
        r"""
        Starts training of the model

        :param feed_dict: dict of values which will be feeded each `session.run()`

            See also feed_dict parameter of `session.run()`.
        :param learning_rate: learning rate used for optimization
        :param convergence_criteria: criteria after which the training will be interrupted.
            Currently implemented criterias:

            - "step":
              stop, when the step counter reaches `stopping_criteria`
            - "difference":
              stop, when `loss(step=i) - loss(step=i-1)` < `stopping_criteria`
            - "moving_average":
              stop, when `mean_loss(steps=[i-2N..i-N) - mean_loss(steps=[i-N..i)` < `stopping_criteria`
            - "absolute_moving_average":
              stop, when `|mean_loss(steps=[i-2N..i-N) - mean_loss(steps=[i-N..i)|` < `stopping_criteria`
            - "t_test" (recommended):
              Perform t-Test between the last [i-2N..i-N] and [i-N..i] losses.
              Stop if P("both distributions are equal") > `stopping_criteria`.
        :param stopping_criteria: Additional parameter for convergence criteria.

            See parameter `convergence_criteria` for exact meaning
        :param loss_window_size: specifies `N` in `convergence_criteria`.
        :param use_batching: If True, will use mini-batches with the batch size defined in the constructor.
            Otherwise, the gradient of the full dataset will be used.
        :param optim_algo: name of the requested train op. Can be:

            - "Adam"
            - "Adagrad"
            - "RMSprop"
            - "GradientDescent" or "GD"

            See :func:train_utils.MultiTrainer.train_op_by_name for further details.
        """
        if use_batching:
            loss = self.model.loss
            train_op = self.model.batch_trainers.train_op_by_name(optim_algo)
        else:
            loss = self.model.full_loss
            train_op = self.model.full_data_trainers.train_op_by_name(optim_algo)

        super().train(*args,
                      feed_dict={"learning_rate:0": learning_rate},
                      convergence_criteria=convergence_criteria,
                      loss_window_size=loss_window_size,
                      stopping_criteria=stopping_criteria,
                      loss=loss,
                      train_op=train_op,
                      **kwargs)

    @property
    def input_data(self) -> InputData:
        return self._input_data

    def train_sequence(self, training_strategy=TrainingStrategy.AUTO):
        if isinstance(training_strategy, Enum):
            training_strategy = training_strategy.value
        elif isinstance(training_strategy, str):
            training_strategy = self.TrainingStrategy[training_strategy].value

        if training_strategy is None:
            training_strategy = self.TrainingStrategy.DEFAULT.value

        logger.info("training strategy:\n%s", pprint.pformat(training_strategy))

        for idx, d in enumerate(training_strategy):
            logger.info("Beginning with training sequence #%d", idx + 1)
            self.train(**d)
            logger.info("Training sequence #%d complete", idx + 1)

    @property
    def a(self):
        return self.to_xarray("a", coords=self.input_data.data.coords)

    @property
    def b(self):
        return self.to_xarray("b", coords=self.input_data.data.coords)

    @property
    def batch_loss(self):
        return self.to_xarray("loss")

    @property
    def batch_gradient(self):
        return self.to_xarray("gradient", coords=self.input_data.data.coords)

    @property
    def loss(self):
        return self.to_xarray("full_loss")

    @property
    def gradient(self):
        return self.to_xarray("full_gradient", coords=self.input_data.data.coords)

    @property
    def design_mixture_loc(self) -> xr.DataArray:
        return self.input_data.design_mixture_loc

    @property
    def design_mixture_scale(self) -> xr.DataArray:
        return self.input_data.design_mixture_scale

    # @property
    # def mixture_prob(self):
    #     return self.run(self.model.mixture_prob)

    @property
    def mixture_log_prob(self):
        return self.to_xarray("mixture_log_prob", coords=self.input_data.data.coords)

    # @property
    # def mixture_logit_prob(self):  # not necessary from model
    #     return self.to_xarray("mixture_logit_prob", coords=self.input_data.data.coords)

    def finalize(self):
        logger.debug("Collect and compute ouptut")
        store = XArrayEstimatorStore(self)
        logger.debug("Closing session")
        self.close_session()
        return store
