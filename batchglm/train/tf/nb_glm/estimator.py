import abc
from typing import Union, Dict, Tuple, List
import logging
import pprint
from enum import Enum

import tensorflow as tf
# import tensorflow_probability as tfp

import numpy as np
from numpy.linalg import matrix_rank

try:
    import anndata
except ImportError:
    anndata = None

from .base import BasicModelGraph, ModelVars, ESTIMATOR_PARAMS
from .base import param_bounds, tf_clip_param, np_clip_param, apply_constraints

from .external import AbstractEstimator, XArrayEstimatorStore, InputData, Model, MonitoredTFEstimator, TFEstimatorGraph
from .external import nb_utils, train_utils, op_utils, rand_utils, data_utils
from .external import pkg_constants
from .hessians import Hessians
from .jacobians import Jacobians

logger = logging.getLogger(__name__)


class FullDataModelGraph:
    def __init__(
            self,
            sample_indices: tf.Tensor,
            fetch_fn,
            batch_size: Union[int, tf.Tensor],
            model_vars,
            constraints_loc,
            constraints_scale,
            dtype
    ):
        num_features = model_vars.a.shape[-1]
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
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                a=model_vars.a,
                b=model_vars.b,
                dtype=dtype,
                size_factors=size_factors)
            return model

        super()
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

        # TODO: remove this and decide for one implementation
        if pkg_constants.HESSIAN_MODE == "obs":
            # Only need iterator that yields single observations for hessian mode obs:
            singleobs_data = dataset.map(fetch_fn, num_parallel_calls=pkg_constants.TF_NUM_THREADS)
            singleobs_data = singleobs_data.prefetch(1)
        else:
            singleobs_data = None

        with tf.name_scope("hessians"):
            hessians = Hessians(
                batched_data=batched_data,
                singleobs_data=singleobs_data,
                sample_indices=sample_indices,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                model_vars=model_vars,
                mode=pkg_constants.HESSIAN_MODE,
                iterator=True,
                dtype=dtype
            )

        with tf.name_scope("jacobians"):
            jacobians = Jacobians(
                batched_data=batched_data,
                sample_indices=sample_indices,
                batch_model=None,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                model_vars=model_vars,
                mode=pkg_constants.JACOBIAN_MODE,
                iterator=True,
                dtype=dtype
            )

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

        self.probs = model.probs
        self.log_probs = model.log_probs

        # custom
        self.sample_indices = sample_indices

        self.log_likelihood = log_likelihood
        self.norm_log_likelihood = norm_log_likelihood
        self.norm_neg_log_likelihood = norm_neg_log_likelihood
        self.loss = loss

        self.jac = jacobians.jac
        self.neg_jac = jacobians.neg_jac
        self.hessian = hessians.hessian
        self.neg_hessian = hessians.neg_hessian


class EstimatorGraph(TFEstimatorGraph):
    X: tf.Tensor

    mu: tf.Tensor
    sigma2: tf.Tensor
    a: tf.Tensor
    b: tf.Tensor

    def __init__(
            self,
            fetch_fn,
            feature_isnonzero,
            num_observations,
            num_features,
            num_design_loc_params,
            num_design_scale_params,
            graph: tf.Graph = None,
            batch_size=500,
            init_a=None,
            init_b=None,
            constraints_loc=None,
            constraints_scale=None,
            extended_summary=False,
            dtype="float32"
    ):
        super().__init__(graph)
        self.num_observations = num_observations
        self.num_features = num_features
        self.num_design_loc_params = num_design_loc_params
        self.num_design_scale_params = num_design_scale_params
        self.batch_size = batch_size

        # initial graph elements
        with self.graph.as_default():
            # ### placeholders
            learning_rate = tf.placeholder(dtype, shape=(), name="learning_rate")
            # train_steps = tf.placeholder(tf.int32, shape=(), name="training_steps")

            # ### performance related settings
            buffer_size = 4

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

            dtype = batch_X.dtype

            # implicit broadcasting of X and initial_mixture_probs to
            #   shape (num_mixtures, num_observations, num_features)
            # init_dist = nb_utils.fit(batch_X, axis=-2)
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
                num_design_loc_params=num_design_loc_params,
                num_design_scale_params=num_design_scale_params,
                num_features=num_features,
                init_a=init_a,
                init_b=init_b,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale
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
                    constraints_loc=constraints_loc,
                    constraints_scale=constraints_scale,
                    a=model_vars.a,
                    b=model_vars.b,
                    dtype=dtype,
                    size_factors=batch_size_factors
                )

                # minimize negative log probability (log(1) = 0);
                # use the mean loss to keep a constant learning rate independently of the batch size
                batch_loss = batch_model.loss

                # Define the jacobian on the batched model for newton-rhapson:
                batch_jac = Jacobians(
                    batched_data=batch_data,
                    sample_indices=batch_sample_index,
                    batch_model=batch_model,
                    constraints_loc=constraints_loc,
                    constraints_scale=constraints_scale,
                    model_vars=model_vars,
                    mode="analytic",
                    iterator=False,
                    dtype=dtype
                )

                # Define the hessian on the batched model for newton-rhapson:
                batch_hessians = Hessians(
                    batched_data=batch_data,
                    singleobs_data=None,
                    sample_indices=batch_sample_index,
                    constraints_loc=constraints_loc,
                    constraints_scale=constraints_scale,
                    model_vars=model_vars,
                    mode="obs_batched",
                    iterator=False,
                    dtype=dtype
                )

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
                    constraints_loc=constraints_loc,
                    constraints_scale=constraints_scale,
                    dtype=dtype,
                )
                full_data_loss = full_data_model.loss
                fisher_inv = op_utils.pinv(full_data_model.neg_hessian)

                # with tf.name_scope("hessian_diagonal"):
                #     hessian_diagonal = [
                #         tf.map_fn(
                #             # elems=tf.transpose(hess, perm=[2, 0, 1]),
                #             elems=hess,
                #             fn=tf.diag_part,
                #             parallel_iterations=pkg_constants.TF_LOOP_PARALLEL_ITERATIONS
                #         )
                #         for hess in full_data_model.hessians
                #     ]
                #     fisher_a, fisher_b = hessian_diagonal

                mu = full_data_model.mu
                r = full_data_model.r
                sigma2 = full_data_model.sigma2

            # ### management
            with tf.name_scope("training"):
                global_step = tf.train.get_or_create_global_step()

                # set up trainers for different selections of variables to train
                # set up multiple optimization algorithms for each trainer
                batch_trainers = train_utils.MultiTrainer(
                    loss=batch_model.norm_neg_log_likelihood,
                    variables=[model_vars.params],
                    learning_rate=learning_rate,
                    global_step=global_step,
                    name="batch_trainers"
                )
                batch_trainers_a_only = train_utils.MultiTrainer(
                    gradients=[
                        (
                            tf.concat([
                                tf.gradients(batch_model.norm_neg_log_likelihood, model_vars.a)[0],
                                tf.zeros_like(model_vars.b),
                            ], axis=0),
                            model_vars.params
                        ),
                    ],
                    learning_rate=learning_rate,
                    global_step=global_step,
                    name="batch_trainers_a_only"
                )
                batch_trainers_b_only = train_utils.MultiTrainer(
                    gradients=[
                        (
                            tf.concat([
                                tf.zeros_like(model_vars.a),
                                tf.gradients(batch_model.norm_neg_log_likelihood, model_vars.b)[0],
                            ], axis=0),
                            model_vars.params
                        ),
                    ],
                    learning_rate=learning_rate,
                    global_step=global_step,
                    name="batch_trainers_b_only"
                )

                with tf.name_scope("batch_gradient"):
                    batch_gradient = batch_trainers.gradient[0][0]
                    batch_gradient = tf.reduce_sum(tf.abs(batch_gradient), axis=0)

                    # batch_gradient = tf.add_n(
                    #     [tf.reduce_sum(tf.abs(grad), axis=0) for (grad, var) in batch_trainers.gradient])

                full_data_trainers = train_utils.MultiTrainer(
                    loss=full_data_model.norm_neg_log_likelihood,
                    variables=[model_vars.params],
                    learning_rate=learning_rate,
                    global_step=global_step,
                    name="full_data_trainers"
                )
                full_data_trainers_a_only = train_utils.MultiTrainer(
                    gradients=[
                        (
                            tf.concat([
                                tf.gradients(full_data_model.norm_neg_log_likelihood, model_vars.a)[0],
                                tf.zeros_like(model_vars.b),
                            ], axis=0),
                            model_vars.params
                        ),
                    ],
                    learning_rate=learning_rate,
                    global_step=global_step,
                    name="full_data_trainers_a_only"
                )
                full_data_trainers_b_only = train_utils.MultiTrainer(
                    gradients=[
                        (
                            tf.concat([
                                tf.zeros_like(model_vars.a),
                                tf.gradients(full_data_model.norm_neg_log_likelihood, model_vars.b)[0],
                            ], axis=0),
                            model_vars.params
                        ),
                    ],
                    learning_rate=learning_rate,
                    global_step=global_step,
                    name="full_data_trainers_b_only"
                )
                with tf.name_scope("full_gradient"):
                    # use same gradient as the optimizers
                    full_gradient = full_data_trainers.gradient[0][0]
                    full_gradient = tf.reduce_sum(tf.abs(full_gradient), axis=0)

                    # # the analytic Jacobian
                    # full_gradient = tf.reduce_sum(full_data_model.neg_jac, axis=0)
                    # full_gradient = tf.add_n(
                    #     [tf.reduce_sum(tf.abs(grad), axis=0) for (grad, var) in full_data_trainers.gradient])

                with tf.name_scope("newton-raphson"):
                    # tf.gradients(- full_data_model.log_likelihood, [model_vars.a, model_vars.b])
                    # Full data model:
                    param_grad_vec = full_data_model.neg_jac
                    # param_grad_vec = tf.gradients(- full_data_model.log_likelihood, model_vars.params)[0]
                    # param_grad_vec_t = tf.transpose(param_grad_vec)

                    delta_t = tf.squeeze(tf.matrix_solve_ls(
                        full_data_model.neg_hessian,
                        # (full_data_model.hessians + tf.transpose(full_data_model.hessians, perm=[0, 2, 1])) / 2, # don't need this with closed forms
                        tf.expand_dims(param_grad_vec, axis=-1),
                        fast=False
                    ), axis=-1)
                    delta = tf.transpose(delta_t)
                    nr_update = model_vars.params - learning_rate * delta
                    # nr_update = model_vars.params - delta
                    newton_raphson_op = tf.group(
                        tf.assign(model_vars.params, nr_update),
                        tf.assign_add(global_step, 1)
                    )

                    # Batched data model:
                    param_grad_vec_batched = batch_jac.neg_jac
                    # param_grad_vec_batched = tf.gradients(- batch_model.log_likelihood,
                    #                                      model_vars.params)[0]
                    # param_grad_vec_batched_t = tf.transpose(param_grad_vec_batched)

                    delta_batched_t = tf.squeeze(tf.matrix_solve_ls(
                        batch_hessians.neg_hessian,
                        tf.expand_dims(param_grad_vec_batched, axis=-1),
                        fast=False
                    ), axis=-1)
                    delta_batched = tf.transpose(delta_batched_t)
                    nr_update_batched = model_vars.params - delta_batched
                    newton_raphson_batched_op = tf.group(
                        tf.assign(model_vars.params, nr_update_batched),
                        tf.assign_add(global_step, 1)
                    )

                # # ### BFGS implementation using SciPy L-BFGS
                # with tf.name_scope("bfgs"):
                #     feature_idx = tf.placeholder(dtype="int64", shape=())
                #
                #     X_s = tf.gather(X, feature_idx, axis=1)
                #     a_s = tf.gather(a, feature_idx, axis=1)
                #     b_s = tf.gather(b, feature_idx, axis=1)
                #
                #     model = BasicModelGraph(X_s, design_loc, design_scale, a_s, b_s, size_factors=size_factors)
                #
                #     trainer = tf.contrib.opt.ScipyOptimizerInterface(
                #         model.loss,
                #         method='L-BFGS-B',
                #         options={'maxiter': maxiter})

            with tf.name_scope("init_op"):
                init_op = tf.global_variables_initializer()

            # ### output values:
            #       override all-zero features with lower bound coefficients
            with tf.name_scope("output"):
                bounds_min, bounds_max = param_bounds(dtype)

                param_nonzero_a = tf.broadcast_to(feature_isnonzero, [num_design_loc_params, num_features])
                alt_a = tf.concat([
                    # intercept
                    tf.broadcast_to(bounds_min["a"], [1, num_features]),
                    # slope
                    tf.zeros(shape=[num_design_loc_params - 1, num_features], dtype=model_vars.a.dtype),
                ], axis=0, name="alt_a")
                a = tf.where(
                    param_nonzero_a,
                    model_vars.a,
                    alt_a,
                    name="a"
                )

                param_nonzero_b = tf.broadcast_to(feature_isnonzero, [num_design_scale_params, num_features])
                alt_b = tf.concat([
                    # intercept
                    tf.broadcast_to(bounds_max["b"], [1, num_features]),
                    # slope
                    tf.zeros(shape=[num_design_scale_params - 1, num_features], dtype=model_vars.b.dtype),
                ], axis=0, name="alt_b")
                b = tf.where(
                    param_nonzero_b,
                    model_vars.b,
                    alt_b,
                    name="b"
                )

        self.fetch_fn = fetch_fn
        self.model_vars = model_vars
        self.batch_model = batch_model

        self.learning_rate = learning_rate
        self.loss = batch_loss

        self.batch_trainers = batch_trainers
        self.batch_trainers_a_only = batch_trainers_a_only
        self.batch_trainers_b_only = batch_trainers_b_only
        self.full_data_trainers = full_data_trainers
        self.full_data_trainers_a_only = full_data_trainers_a_only
        self.full_data_trainers_b_only = full_data_trainers_b_only
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
        assert (self.a.shape == (num_design_loc_params, num_features))
        assert (self.b.shape == (num_design_scale_params, num_features))

        self.mu = mu
        self.r = r
        self.sigma2 = sigma2

        self.batch_probs = batch_model.probs
        self.batch_log_probs = batch_model.log_probs
        self.batch_log_likelihood = batch_model.norm_log_likelihood

        self.sample_selection = sample_selection
        self.full_data_model = full_data_model

        self.full_loss = full_data_loss

        self.full_gradient = full_gradient
        # self.full_gradient_a = full_gradient_a
        # self.full_gradient_b = full_gradient_b

        self.hessians = full_data_model.hessian
        self.fisher_inv = fisher_inv

        self.newton_raphson_op = newton_raphson_op
        self.newton_raphson_batched_op = newton_raphson_batched_op

        with tf.name_scope('summaries'):
            tf.summary.histogram('a', model_vars.a)
            tf.summary.histogram('b', model_vars.b)
            tf.summary.scalar('loss', batch_loss)
            tf.summary.scalar('learning_rate', learning_rate)

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

        self.saver = tf.train.Saver()
        self.merged_summary = tf.summary.merge_all()


class Estimator(AbstractEstimator, MonitoredTFEstimator, metaclass=abc.ABCMeta):
    """
    Estimator for Generalized Linear Models (GLMs) with negative binomial noise.
    Uses the natural logarithm as linker function.
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
        NEWTON_EXACT = [
            {
                "learning_rate": 1,
                "convergence_criteria": "scaled_moving_average",
                "stopping_criteria": 1e-8,
                "loss_window_size": 5,
                "use_batching": False,
                "optim_algo": "newton-raphson",
            },
        ]
        NEWTON_BATCHED = [
            {
                "learning_rate": 1,
                "convergence_criteria": "scaled_moving_average",
                "stopping_criteria": 1e-8,
                "loss_window_size": 20,
                "use_batching": True,
                "optim_algo": "newton-raphson",
            },
        ]
        NEWTON_SERIES = [
            {
                "learning_rate": 1,
                "convergence_criteria": "scaled_moving_average",
                "stopping_criteria": 1e-8,
                "loss_window_size": 8,
                "use_batching": True,
                "optim_algo": "newton-raphson",
            },
            {
                "learning_rate": 1,
                "convergence_criteria": "scaled_moving_average",
                "stopping_criteria": 1e-8,
                "loss_window_size": 4,
                "use_batching": False,
                "optim_algo": "newton-raphson",
            },
        ]

    model: EstimatorGraph
    _train_mu: bool
    _train_r: bool

    @classmethod
    def param_shapes(cls) -> dict:
        return ESTIMATOR_PARAMS

    def __init__(
            self,
            input_data: InputData,
            batch_size: int = 500,
            init_model: Model = None,
            graph: tf.Graph = None,
            init_a: Union[np.ndarray, str] = "AUTO",
            init_b: Union[np.ndarray, str] = "AUTO",
            quick_scale: bool = False,
            model: EstimatorGraph = None,
            extended_summary=False,
            dtype="float64",
    ):
        """
        Create a new Estimator

        :param input_data: The input data
        :param batch_size: The batch size to use for minibatch SGD.
            Defaults to '500'
        :param graph: (optional) tf.Graph
        :param init_model: (optional) If provided, this model will be used to initialize this Estimator.
        :param init_a: (Optional) Low-level initial values for a.
            Can be:

            - str:
                * "auto": automatically choose best initialization
                * "random": initialize with random values
                * "standard": initialize intercept with observed mean
                * "init_model": initialize with another model (see `ìnit_model` parameter)
                * "closed_form": try to initialize with closed form
            - np.ndarray: direct initialization of 'a'
        :param init_b: (Optional) Low-level initial values for b
            Can be:

            - str:
                * "auto": automatically choose best initialization
                * "random": initialize with random values
                * "standard": initialize with zeros
                * "init_model": initialize with another model (see `ìnit_model` parameter)
                * "closed_form": try to initialize with closed form
            - np.ndarray: direct initialization of 'b'
        :param model: (optional) EstimatorGraph to use. Basically for debugging.
        :param quick_scale: `scale` will be fitted faster and maybe less accurate.

        Useful in scenarios where fitting the exact `scale` is not absolutely necessary.
        :param extended_summary: Include detailed information in the summaries.
            Will drastically increase runtime of summary writer, use only for debugging.
        """
        # validate design matrix:
        if np.linalg.matrix_rank(input_data.design_loc) != np.linalg.matrix_rank(input_data.design_loc.T):
            raise ValueError("design_loc matrix is not full rank")
        if np.linalg.matrix_rank(input_data.design_scale) != np.linalg.matrix_rank(input_data.design_scale.T):
            raise ValueError("design_scale matrix is not full rank")

        # ### initialization
        if model is None:
            if graph is None:
                graph = tf.Graph()

            self._input_data = input_data
            self._train_mu = True
            self._train_r = not quick_scale

            r"""
            standard:
            Only initialise intercept and keep other coefficients as zero.

            closed-form:
            Initialize with Maximum Likelihood / Maximum of Momentum estimators

            Idea:
            $$
                \theta &= f(x) \\
                \Rightarrow f^{-1}(\theta) &= x \\
                    &= (D \cdot D^{+}) \cdot x \\
                    &= D \cdot (D^{+} \cdot x) \\
                    &= D \cdot x' = f^{-1}(\theta)
            $$
            """
            size_factors_init = input_data.size_factors
            if size_factors_init is not None:
                size_factors_init = np.expand_dims(size_factors_init, axis=1)
                size_factors_init = np.broadcast_to(
                    array=size_factors_init,
                    shape=[input_data.num_observations, input_data.num_features]
                )

            groupwise_means = None  # [groups, features]
            overall_means = None  # [1, features]
            logger.debug(" * Initialize mean model")
            if isinstance(init_a, str):
                # Chose option if auto was chosen
                if init_a.lower() == "auto":
                    init_a = "closed_form"

                if init_a.lower() == "closed_form":
                    try:
                        unique_design_loc, inverse_idx = np.unique(input_data.design_loc, axis=0, return_inverse=True)
                        if input_data.constraints_loc is not None:
                            unique_design_loc_constraints = input_data.constraints_loc.copy()
                            # -1 in the constraint matrix is used to indicate which variable
                            # is made dependent so that the constrained is fullfilled.
                            # This has to be rewritten here so that the design matrix is full rank
                            # which is necessary so that it can be inverted for parameter
                            # initialisation.
                            unique_design_loc_constraints[unique_design_loc_constraints == -1] = 1
                            # Add constraints into design matrix to remove structural unidentifiability.
                            unique_design_loc = np.vstack([unique_design_loc, unique_design_loc_constraints])

                        if unique_design_loc.shape[1] > matrix_rank(unique_design_loc):
                            logger.warning("Location model is not full rank!")
                        X = input_data.X.assign_coords(group=(("observations",), inverse_idx))
                        if size_factors_init is not None:
                            X = np.divide(X, size_factors_init)

                        groupwise_means = X.groupby("group").mean(dim="observations").values
                        # clipping
                        groupwise_means = np_clip_param(groupwise_means, "mu")
                        # mean = np.nextafter(0, 1, out=mean.values, where=mean == 0, dtype=mean.dtype)

                        a = np.log(groupwise_means)
                        if input_data.constraints_loc is not None:
                            a_constraints = np.zeros([input_data.constraints_loc.shape[0], a.shape[1]])
                            # Add constraints (sum to zero) to value vector to remove structural unidentifiability.
                            a = np.vstack([a, a_constraints])

                        # inv_design = np.linalg.pinv(unique_design_loc) # NOTE: this is numerically inaccurate!
                        # inv_design = np.linalg.inv(unique_design_loc) # NOTE: this is exact if full rank!
                        # init_a = np.matmul(inv_design, a)
                        #
                        # Use least-squares solver to calculate a':
                        # This is faster and more accurate than using matrix inversion.
                        logger.debug(" ** Solve lstsq problem")
                        a_prime = np.linalg.lstsq(unique_design_loc, a, rcond=None)
                        init_a = a_prime[0]
                        # stat_utils.rmsd(np.exp(unique_design_loc @ init_a), mean)

                        # train mu, if the closed-form solution is inaccurate
                        self._train_mu = not np.all(a_prime[1] == 0)

                        # Temporal fix: train mu if size factors are given as closed form may be different:
                        if input_data.size_factors is not None:
                            self._train_mu = True

                        logger.info("Using closed-form MLE initialization for mean")
                        logger.debug("RMSE of closed-form mean:\n%s", a_prime[1])
                        logger.info("Should train mu: %s", self._train_mu)
                    except np.linalg.LinAlgError:
                        logger.warning("Closed form initialization failed!")
                elif init_a.lower() == "standard":
                    overall_means = input_data.X.mean(dim="observations").values  # directly calculate the mean
                    # clipping
                    overall_means = np_clip_param(overall_means, "mu")
                    # mean = np.nextafter(0, 1, out=mean, where=mean == 0, dtype=mean.dtype)

                    init_a = np.zeros([input_data.num_design_loc_params, input_data.num_features])
                    init_a[0, :] = np.log(overall_means)
                    self._train_mu = True

                    logger.info("Using standard initialization for mean")
                    logger.info("Should train mu: %s", self._train_mu)

            logger.debug(" * Initialize dispersion model")
            if isinstance(init_b, str):
                if init_b.lower() == "auto":
                    init_b = "closed_form"

                if init_b.lower() == "closed_form":
                    try:
                        unique_design_scale, inverse_idx = np.unique(input_data.design_scale, axis=0,
                                                                     return_inverse=True)
                        if input_data.constraints_scale is not None:
                            unique_design_scale_constraints = input_data.constraints_scale.copy()
                            # -1 in the constraint matrix is used to indicate which variable
                            # is made dependent so that the constrained is fullfilled.
                            # This has to be rewritten here so that the design matrix is full rank
                            # which is necessary so that it can be inverted for parameter
                            # initialisation.
                            unique_design_scale_constraints[unique_design_scale_constraints == -1] = 1
                            # Add constraints into design matrix to remove structural unidentifiability.
                            unique_design_scale = np.vstack([unique_design_scale, unique_design_scale_constraints])

                        if unique_design_scale.shape[1] > matrix_rank(unique_design_scale):
                            logger.warning("Scale model is not full rank!")

                        X = input_data.X.assign_coords(group=(("observations",), inverse_idx))
                        if input_data.size_factors is not None:
                            X = np.divide(X, size_factors_init)

                        # Xdiff = X - np.exp(input_data.design_loc @ init_a)
                        # Define xarray version of init so that Xdiff can be evaluated lazy by dask.
                        init_a_xr = data_utils.xarray_from_data(init_a, dims=("design_loc_params", "features"))
                        init_a_xr.coords["design_loc_params"] = input_data.design_loc.coords["design_loc_params"]
                        logger.debug(" ** Define Xdiff")
                        Xdiff = X - np.exp(input_data.design_loc.dot(init_a_xr))
                        variance = np.square(Xdiff).groupby("group").mean(dim="observations")

                        if groupwise_means is None:
                            groupwise_means = X.groupby("group").mean(dim="observations")
                        denominator = np.fmax(variance - groupwise_means, 0)
                        denominator = np.nextafter(0, 1, out=denominator.values, where=denominator == 0,
                                                   dtype=denominator.dtype)
                        r = np.asarray(np.square(groupwise_means) / denominator)
                        # clipping
                        r = np_clip_param(r, "r")
                        # r = np.nextafter(0, 1, out=r.values, where=r == 0, dtype=r.dtype)
                        # r = np.fmin(r, np.finfo(r.dtype).max)

                        b = np.log(r)
                        if input_data.constraints_scale is not None:
                            b_constraints = np.zeros([input_data.constraints_scale.shape[0], b.shape[1]])
                            # Add constraints (sum to zero) to value vector to remove structural unidentifiability.
                            b = np.vstack([b, b_constraints])

                        # inv_design = np.linalg.pinv(unique_design_scale) # NOTE: this is numerically inaccurate!
                        # inv_design = np.linalg.inv(unique_design_scale) # NOTE: this is exact if full rank!
                        # init_b = np.matmul(inv_design, b)
                        #
                        # Use least-squares solver to calculate a':
                        # This is faster and more accurate than using matrix inversion.
                        logger.debug(" ** Solve lstsq problem")
                        b_prime = np.linalg.lstsq(unique_design_scale, b, rcond=None)
                        init_b = b_prime[0]

                        logger.info("Using closed-form MME initialization for dispersion")
                        logger.debug("RMSE of closed-form dispersion:\n%s", b_prime[1])
                        logger.info("Should train r: %s", self._train_r)
                    except np.linalg.LinAlgError:
                        logger.warning("Closed form initialization failed!")
                elif init_b.lower() == "standard":
                    init_b = np.zeros([input_data.design_scale.shape[1], input_data.X.shape[1]])

                    logger.info("Using standard initialization for dispersion")
                    logger.info("Should train r: %s", self._train_r)

            if init_model is not None:
                if isinstance(init_a, str) and (init_a.lower() == "auto" or init_a.lower() == "init_model"):
                    # location
                    my_loc_names = set(input_data.design_loc_names.values)
                    my_loc_names = my_loc_names.intersection(init_model.input_data.design_loc_names.values)

                    init_loc = np.random.uniform(
                        low=np.nextafter(0, 1, dtype=input_data.X.dtype),
                        high=np.sqrt(np.nextafter(0, 1, dtype=input_data.X.dtype)),
                        size=(input_data.num_design_loc_params, input_data.num_features)
                    )
                    for parm in my_loc_names:
                        init_idx = np.where(init_model.input_data.design_loc_names == parm)
                        my_idx = np.where(input_data.design_loc_names == parm)
                        init_loc[my_idx] = init_model.par_link_loc[init_idx]

                    init_a = init_loc

                if isinstance(init_b, str) and (init_b.lower() == "auto" or init_b.lower() == "init_model"):
                    # scale
                    my_scale_names = set(input_data.design_scale_names.values)
                    my_scale_names = my_scale_names.intersection(init_model.input_data.design_scale_names.values)

                    init_scale = np.random.uniform(
                        low=np.nextafter(0, 1, dtype=input_data.X.dtype),
                        high=np.sqrt(np.nextafter(0, 1, dtype=input_data.X.dtype)),
                        size=(input_data.num_design_scale_params, input_data.num_features)
                    )
                    for parm in my_scale_names:
                        init_idx = np.where(init_model.input_data.design_scale_names == parm)
                        my_idx = np.where(input_data.design_scale_names == parm)
                        init_scale[my_idx] = init_model.par_link_scale[init_idx]

                    init_b = init_scale

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

        logger.debug(" * Start creating model")
        with graph.as_default():
            # create model
            model = EstimatorGraph(
                fetch_fn=fetch_fn,
                feature_isnonzero=input_data.feature_isnonzero,
                num_observations=input_data.num_observations,
                num_features=input_data.num_features,
                num_design_loc_params=input_data.num_design_loc_params,
                num_design_scale_params=input_data.num_design_scale_params,
                batch_size=batch_size,
                graph=graph,
                init_a=init_a,
                init_b=init_b,
                constraints_loc=input_data.constraints_loc,
                constraints_scale=input_data.constraints_scale,
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
              train_mu: bool = None,
              train_r: bool = None,
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
        :param train_mu: Set to True/False in order to enable/disable training of mu
        :param train_r: Set to True/False in order to enable/disable training of r
        :param use_batching: If True, will use mini-batches with the batch size defined in the constructor.
            Otherwise, the gradient of the full dataset will be used.
        :param optim_algo: name of the requested train op. Can be:

            - "Adam"
            - "Adagrad"
            - "RMSprop"
            - "GradientDescent" or "GD"

            See :func:train_utils.MultiTrainer.train_op_by_name for further details.
        """
        if train_mu is None:
            # check if mu was initialized with MLE
            train_mu = self._train_mu
        if train_r is None:
            # check if r was initialized with MLE
            train_r = self._train_r

        if use_batching:
            loss = self.model.loss
            if optim_algo.lower() == "newton" or \
                    optim_algo.lower() == "newton-raphson" or \
                    optim_algo.lower() == "newton_raphson" or \
                    optim_algo.lower() == "nr":
                train_op = self.model.newton_raphson_batched_op
            elif train_mu:
                if train_r:
                    train_op = self.model.batch_trainers.train_op_by_name(optim_algo)
                else:
                    train_op = self.model.batch_trainers_a_only.train_op_by_name(optim_algo)
            else:
                if train_r:
                    train_op = self.model.batch_trainers_b_only.train_op_by_name(optim_algo)
                else:
                    logger.info("No training necessary; returning")
                    return
        else:
            loss = self.model.full_loss
            if optim_algo.lower() == "newton" or \
                    optim_algo.lower() == "newton-raphson" or \
                    optim_algo.lower() == "newton_raphson" or \
                    optim_algo.lower() == "nr":
                train_op = self.model.newton_raphson_op
            elif train_mu:
                if train_r:
                    train_op = self.model.full_data_trainers.train_op_by_name(optim_algo)
                else:
                    train_op = self.model.full_data_trainers_a_only.train_op_by_name(optim_algo)
            else:
                if train_r:
                    train_op = self.model.full_data_trainers_b_only.train_op_by_name(optim_algo)
                else:
                    logger.info("No training necessary; returning")
                    return

        super().train(*args,
                      feed_dict={"learning_rate:0": learning_rate},
                      convergence_criteria=convergence_criteria,
                      loss_window_size=loss_window_size,
                      stopping_criteria=stopping_criteria,
                      loss=loss,
                      train_op=train_op,
                      **kwargs)

    @property
    def input_data(self):
        return self._input_data

    def train_sequence(self, training_strategy=TrainingStrategy.AUTO):
        if isinstance(training_strategy, Enum):
            training_strategy = training_strategy.value
        elif isinstance(training_strategy, str):
            training_strategy = self.TrainingStrategy[training_strategy].value

        if training_strategy is None:
            if not self._train_mu:
                training_strategy = self.TrainingStrategy.PRE_INITIALIZED.value
            else:
                training_strategy = self.TrainingStrategy.DEFAULT.value

        logger.info("training strategy:\n%s", pprint.pformat(training_strategy))

        for idx, d in enumerate(training_strategy):
            logger.info("Beginning with training sequence #%d", idx + 1)
            self.train(**d)
            logger.info("Training sequence #%d complete", idx + 1)

    # @property
    # def mu(self):
    #     return self.to_xarray("mu")
    #
    # @property
    # def r(self):
    #     return self.to_xarray("r")
    #
    # @property
    # def sigma2(self):
    #     return self.to_xarray("sigma2")

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
    def hessians(self):
        return self.to_xarray("hessians", coords=self.input_data.data.coords)

    @property
    def fisher_inv(self):
        return self.to_xarray("fisher_inv", coords=self.input_data.data.coords)

    def finalize(self):
        logger.debug("Collect and compute ouptut")
        store = XArrayEstimatorStore(self)
        logger.debug("Closing session")
        self.close_session()
        return store
