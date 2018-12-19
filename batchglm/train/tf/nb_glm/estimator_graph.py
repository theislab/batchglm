from typing import Union
import logging

import tensorflow as tf

import numpy as np

try:
    import anndata
except ImportError:
    anndata = None

from .base import BasicModelGraph, ModelVars
from .base import param_bounds

from .external import TFEstimatorGraph
from .external import nb_utils, train_utils, op_utils
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

        with tf.name_scope("hessians"):
            hessians = Hessians(
                batched_data=batched_data,
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
            train_mu=True,
            train_r=True,
            provide_optimizers: dict = {"gd": True, "adam": True, "adagrad": True, "rmsprop": True, "nr": True},
            termination_type: str = "global",
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
                logger.debug(" ** Build input pipeline")
                data_indices = tf.data.Dataset.from_tensor_slices((
                    tf.range(num_observations, name="sample_index")
                ))
                training_data = data_indices.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=2 * batch_size))
                training_data = training_data.batch(batch_size, drop_remainder=True)
                training_data = training_data.map(tf.contrib.framework.sort)  # sort indices
                training_data = training_data.map(fetch_fn, num_parallel_calls=pkg_constants.TF_NUM_THREADS)
                training_data = training_data.prefetch(buffer_size)

                iterator = training_data.make_one_shot_iterator()

                batch_sample_index, batch_data = iterator.get_next()
                (batch_X, batch_design_loc, batch_design_scale, batch_size_factors) = batch_data

            # dtype = batch_X.dtype

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
                logger.debug(" ** Build batched data model")
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
                    mode=pkg_constants.JACOBIAN_MODE,  #"analytic",
                    iterator=False,
                    dtype=dtype
                )

                # Define the hessian on the batched model for newton-rhapson:
                batch_hessians = Hessians(
                    batched_data=batch_data,
                    sample_indices=batch_sample_index,
                    constraints_loc=constraints_loc,
                    constraints_scale=constraints_scale,
                    model_vars=model_vars,
                    mode=pkg_constants.HESSIAN_MODE,  #"obs_batched",
                    iterator=False,
                    dtype=dtype
                )

            with tf.name_scope("full_data"):
                logger.debug(" ** Build full data model")
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
                logger.debug(" ** Build training graphs")
                global_step = tf.train.get_or_create_global_step()
                idx_nonconverged = np.where(model_vars.converged == False)[0]

                # Set up trainers for different selections of variables to train.
                # Set up multiple optimization algorithms for each trainer.
                # Note that params is tf.Variable and a, b are tensors as they are
                # slices of a variable! Accordingly, the updates are implemented differently.

                if train_mu:
                    if train_r:
                        if termination_type == "by_feature":
                            if provide_optimizers["nr"]:
                                logger.debug(" ** Build newton training graph: by_feature, train mu and r")
                                # Full data model by gene:
                                param_grad_vec = full_data_model.neg_jac
                                # Compute parameter update for non-converged gene only.
                                delta_t_bygene_nonconverged = tf.squeeze(tf.matrix_solve_ls(
                                    tf.gather(
                                        full_data_model.neg_hessian,
                                        indices=idx_nonconverged,
                                        axis=0),
                                    tf.expand_dims(
                                        tf.gather(
                                            param_grad_vec,
                                            indices=idx_nonconverged,
                                            axis=0)
                                        , axis=-1),
                                    fast=False
                                ), axis=-1)
                                # Write parameter updates into matrix of size of all parameters which
                                # contains zero entries for updates of already converged genes.
                                delta_t_bygene_full = tf.concat([
                                    tf.gather(delta_t_bygene_nonconverged,
                                              indices=np.where(idx_nonconverged == i)[0],
                                              axis=0)
                                    if not model_vars.converged[i]
                                    else tf.zeros([1, model_vars.params.shape[0]])
                                    for i in range(model_vars.n_features)
                                ], axis=0)
                                delta_bygene_full = tf.transpose(delta_t_bygene_full)
                                nr_update = model_vars.params - learning_rate * delta_bygene_full

                                # Batched data model by gene:
                                param_grad_vec_batched = batch_jac.neg_jac
                                # Compute parameter update for non-converged gene only.
                                delta_batched_t_bygene_nonconverged = tf.squeeze(tf.matrix_solve_ls(
                                    tf.gather(
                                        batch_hessians.neg_hessian,
                                        indices=idx_nonconverged,
                                        axis=0),
                                    tf.expand_dims(
                                        tf.gather(
                                            param_grad_vec_batched,
                                            indices=idx_nonconverged,
                                            axis=0)
                                        , axis=-1),
                                    fast=False
                                ), axis=-1)
                                # Write parameter updates into matrix of size of all parameters which
                                # contains zero entries for updates of already converged genes.
                                delta_batched_t_bygene_full = tf.concat([
                                    tf.gather(delta_batched_t_bygene_nonconverged,
                                              indices=np.where(idx_nonconverged == i)[0],
                                              axis=0)
                                    if not model_vars.converged[i]
                                    else tf.zeros([1, model_vars.params.shape[0]])
                                    for i in range(model_vars.n_features)
                                ], axis=0)
                                delta_batched_bygene_full = tf.transpose(delta_batched_t_bygene_full)
                                nr_update_batched = model_vars.params - learning_rate * delta_batched_bygene_full
                            else:
                                nr_update = None
                                nr_update_batched = None

                            logger.debug(" ** Build training graph: by_feature, train mu and r")
                            logger.debug(" *** Build training graph: full data")
                            logger.debug(" **** Build gradient graph")
                            gradients_batch = [
                                    (
                                        tf.concat([
                                            tf.gradients(batch_model.norm_neg_log_likelihood,
                                                         model_vars.params_by_gene[i])[0]
                                            if not model_vars.converged[i]
                                            else tf.zeros([model_vars.params.shape[0], 1], dtype=dtype)
                                            for i in range(model_vars.params.shape[1])
                                        ], axis=1),
                                        model_vars.params
                                    )
                                ]
                            trainer_batch = train_utils.MultiTrainer(
                                variables=[model_vars.params],
                                gradients=gradients_batch,
                                newton_delta=nr_update_batched,
                                learning_rate=learning_rate,
                                global_step=global_step,
                                apply_gradients=lambda grad: tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad),
                                provide_optimizers=provide_optimizers,
                                name="batch_trainers_bygene"
                            )
                            logger.debug(" *** Build training graph: batched data")
                            logger.debug(" **** Build gradient graph")
                            gradients_full = [
                                    (
                                        tf.concat([
                                            tf.gradients(full_data_model.norm_neg_log_likelihood,
                                                         model_vars.params_by_gene[i])[0]
                                            if not model_vars.converged[i]
                                            else tf.zeros([model_vars.params.shape[0], 1], dtype=dtype)
                                            for i in range(model_vars.params.shape[1])
                                        ], axis=1),
                                        model_vars.params
                                    )
                                ]
                            trainer_full = train_utils.MultiTrainer(
                                variables=[model_vars.params],
                                gradients=gradients_full,
                                newton_delta=nr_update,
                                learning_rate=learning_rate,
                                global_step=global_step,
                                apply_gradients=lambda grad: tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad),
                                provide_optimizers=provide_optimizers,
                                name="full_data_trainers_bygene"
                            )
                        elif termination_type == "global":
                            if provide_optimizers["nr"]:
                                logger.debug(" ** Build newton training graph: global, train mu and r")
                                # Full data model:
                                param_grad_vec = full_data_model.neg_jac
                                delta_t = tf.squeeze(tf.matrix_solve_ls(
                                    full_data_model.neg_hessian,
                                    # (full_data_model.hessians + tf.transpose(full_data_model.hessians, perm=[0, 2, 1])) / 2,  # symmetrization, don't need this with closed forms
                                    tf.expand_dims(param_grad_vec, axis=-1),
                                    fast=False
                                ), axis=-1)
                                delta = tf.transpose(delta_t)
                                nr_update = model_vars.params - learning_rate * delta

                                # Batched data model:
                                param_grad_vec_batched = batch_jac.neg_jac
                                delta_batched_t = tf.squeeze(tf.matrix_solve_ls(
                                    batch_hessians.neg_hessian,
                                    tf.expand_dims(param_grad_vec_batched, axis=-1),
                                    fast=False
                                ), axis=-1)
                                delta_batched = tf.transpose(delta_batched_t)
                                nr_update_batched = model_vars.params - delta_batched
                            else:
                                nr_update = None
                                nr_update_batched = None

                            logger.debug(" ** Build training graph: global, train mu and r")
                            logger.debug(" *** Build training graph: batched data")
                            trainer_batch = train_utils.MultiTrainer(
                                loss=batch_model.norm_neg_log_likelihood,
                                variables=[model_vars.params],
                                newton_delta=nr_update_batched,
                                learning_rate=learning_rate,
                                global_step=global_step,
                                apply_gradients=lambda grad: tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad),
                                provide_optimizers=provide_optimizers,
                                name="batch_trainers"
                            )
                            logger.debug(" *** Build training graph: full data")
                            trainer_full = train_utils.MultiTrainer(
                                loss=full_data_model.norm_neg_log_likelihood,
                                variables=[model_vars.params],
                                newton_delta=nr_update,
                                learning_rate=learning_rate,
                                global_step=global_step,
                                apply_gradients=lambda grad: tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad),
                                provide_optimizers=provide_optimizers,
                                name="full_data_trainers"
                            )
                        else:
                            raise ValueError("convergence_type %s not recognized." % termination_type)
                    else:
                        if termination_type == "by_feature":
                            logger.debug(" ** Build training graph: by_feature, train mu only")
                            logger.debug(" *** Build training graph: batched data")
                            logger.debug(" **** Build gradient graph")
                            gradients_batch = [
                                    (
                                        tf.concat([
                                            tf.concat([
                                                tf.gradients(batch_model.norm_neg_log_likelihood,
                                                             model_vars.a_by_gene[i])[0]
                                                if not model_vars.converged[i]
                                                else tf.zeros([model_vars.a.shape[0], 1], dtype=dtype)
                                                for i in range(model_vars.a.shape[1])
                                            ], axis=1),
                                            tf.zeros_like(model_vars.b)
                                        ], axis=0),
                                        model_vars.params
                                    ),
                                ]
                            trainer_batch = train_utils.MultiTrainer(
                                gradients=gradients_batch,
                                learning_rate=learning_rate,
                                global_step=global_step,
                                apply_gradients=lambda grad: tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad),
                                provide_optimizers=provide_optimizers,
                                name="batch_trainers_a_only_bygene"
                            )
                            logger.debug(" *** Build training graph: full data")
                            logger.debug(" **** Build gradient graph")
                            gradients_full = [
                                    (
                                        tf.concat([
                                            tf.concat([
                                                tf.gradients(full_data_model.norm_neg_log_likelihood,
                                                             model_vars.a_by_gene[i])[0]
                                                if not model_vars.converged[i]
                                                else tf.zeros([model_vars.a.shape[0], 1], dtype=dtype)
                                                for i in range(model_vars.a.shape[1])
                                            ], axis=1),
                                            tf.zeros_like(model_vars.b)
                                        ], axis=0),
                                        model_vars.params
                                    ),
                                ]
                            trainer_full = train_utils.MultiTrainer(
                                gradients=gradients_full,
                                learning_rate=learning_rate,
                                global_step=global_step,
                                apply_gradients=lambda grad: tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad),
                                provide_optimizers=provide_optimizers,
                                name="full_data_trainers_a_only_bygene"
                            )
                        elif termination_type == "global":
                            logger.debug(" ** Build training graph: global, train mu only")
                            logger.debug(" *** Build training graph: batched data")
                            logger.debug(" **** Build gradient graph")
                            gradients_batch = [
                                    (
                                        tf.concat([
                                            tf.gradients(batch_model.norm_neg_log_likelihood,
                                                         model_vars.a)[0],
                                            tf.zeros_like(model_vars.b, dtype=dtype),
                                        ], axis=0),
                                        model_vars.params
                                    ),
                                ]
                            trainer_batch = train_utils.MultiTrainer(
                                gradients=gradients_batch,
                                learning_rate=learning_rate,
                                global_step=global_step,
                                apply_gradients=lambda grad: tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad),
                                provide_optimizers=provide_optimizers,
                                name="batch_trainers_a_only"
                            )
                            logger.debug(" *** Build training graph: full data")
                            logger.debug(" **** Build gradient graph")
                            gradients_full = [
                                    (
                                        tf.concat([
                                            tf.gradients(full_data_model.norm_neg_log_likelihood,
                                                         model_vars.a)[0],
                                            tf.zeros_like(model_vars.b),
                                        ], axis=0),
                                        model_vars.params
                                    ),
                                ]
                            trainer_full = full_data_trainers_a_only = train_utils.MultiTrainer(
                                gradients=gradients_full,
                                learning_rate=learning_rate,
                                global_step=global_step,
                                apply_gradients=lambda grad: tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad),
                                provide_optimizers=provide_optimizers,
                                name="full_data_trainers_a_only"
                            )
                        else:
                            raise ValueError("convergence_type %s not recognized." % termination_type)
                elif train_r:
                    if termination_type == "by_feature":
                        logger.debug(" ** Build training graph: by_feature, train r only")
                        logger.debug(" *** Build training graph: batched data")
                        logger.debug(" **** Build gradient graph")
                        gradients_batch = [
                                (
                                    tf.concat([
                                        tf.zeros_like(model_vars.a),
                                        tf.concat([
                                            tf.gradients(batch_model.norm_neg_log_likelihood,
                                                         model_vars.b_by_gene[i])[0]
                                            if not model_vars.converged[i]
                                            else tf.zeros([model_vars.b.shape[0], 1], dtype=dtype)
                                            for i in range(model_vars.b.shape[1])
                                        ], axis=1)
                                    ], axis=0),
                                    model_vars.params
                                ),
                            ]
                        trainer_batch = train_utils.MultiTrainer(
                            gradients=gradients_batch,
                            learning_rate=learning_rate,
                            global_step=global_step,
                            apply_gradients=lambda grad: tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad),
                            provide_optimizers=provide_optimizers,
                            name="batch_trainers_b_only_bygene"
                        )
                        logger.debug(" *** Build training graph: full data")
                        logger.debug(" **** Build gradient graph")
                        gradients_full = [
                                (
                                    tf.concat([
                                        tf.zeros_like(model_vars.a),
                                        tf.concat([
                                            tf.gradients(full_data_model.norm_neg_log_likelihood,
                                                         model_vars.b_by_gene[i])[0]
                                            if not model_vars.converged[i]
                                            else tf.zeros([model_vars.b.shape[0], 1], dtype=dtype)
                                            for i in range(model_vars.b.shape[1])
                                        ], axis=1)
                                    ], axis=0),
                                    model_vars.params
                                ),
                            ]
                        trainer_full = train_utils.MultiTrainer(
                            gradients=gradients_full,
                            learning_rate=learning_rate,
                            global_step=global_step,
                            apply_gradients=lambda grad: tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad),
                            provide_optimizers=provide_optimizers,
                            name="full_data_trainers_b_only_bygene"
                        )
                    elif termination_type == "global":
                        logger.debug(" ** Build training graph: global, train r only")
                        logger.debug(" *** Build training graph: batched data")
                        logger.debug(" **** Build gradient graph")
                        gradients_batch = [
                                (
                                    tf.concat([
                                        tf.zeros_like(model_vars.a),
                                        tf.gradients(batch_model.norm_neg_log_likelihood,
                                                     model_vars.b)[0],
                                    ], axis=0),
                                    model_vars.params
                                ),
                            ]
                        trainer_batch = train_utils.MultiTrainer(
                            gradients=gradients_batch,
                            learning_rate=learning_rate,
                            global_step=global_step,
                            apply_gradients=lambda grad: tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad),
                            provide_optimizers=provide_optimizers,
                            name="batch_trainers_b_only"
                        )
                        logger.debug(" *** Build training graph: full data")
                        logger.debug(" **** Build gradient graph")
                        gradients_full = [
                                (
                                    tf.concat([
                                        tf.zeros_like(model_vars.a),
                                        tf.gradients(full_data_model.norm_neg_log_likelihood,
                                                     model_vars.b)[0],
                                    ], axis=0),
                                    model_vars.params
                                ),
                            ]
                        trainer_full = train_utils.MultiTrainer(
                            gradients=gradients_full,
                            learning_rate=learning_rate,
                            global_step=global_step,
                            apply_gradients=lambda grad: tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad),
                            provide_optimizers=provide_optimizers,
                            name="full_data_trainers_b_only"
                        )
                    else:
                        raise ValueError("convergence_type %s not recognized." % termination_type)
                else:
                    logger.info("No training necessary; returning")
                    return False

                # Set up model gradient computation:
                with tf.name_scope("batch_gradient"):
                    logger.debug(" ** Build training graph: batched data model gradients")
                    batch_gradient = trainer_batch.plain_gradient_by_variable(model_vars.params)
                    batch_gradient = tf.reduce_sum(tf.abs(batch_gradient), axis=0)

                    # batch_gradient = tf.add_n(
                    #     [tf.reduce_sum(tf.abs(grad), axis=0) for (grad, var) in batch_trainers.gradient])

                with tf.name_scope("full_gradient"):
                    logger.debug(" ** Build training graph: full data model gradients")
                    # use same gradient as the optimizers
                    full_gradient = trainer_full.plain_gradient_by_variable(model_vars.params)
                    full_gradient = tf.reduce_sum(tf.abs(full_gradient), axis=0)

                    # # the analytic Jacobian
                    # full_gradient = tf.reduce_sum(full_data_model.neg_jac, axis=0)
                    # full_gradient = tf.add_n(
                    #     [tf.reduce_sum(tf.abs(grad), axis=0) for (grad, var) in full_data_trainers.gradient])

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
                logger.debug(" ** Build training graph: initialization operation")
                init_op = tf.global_variables_initializer()

            # ### output values:
            #       override all-zero features with lower bound coefficients
            with tf.name_scope("output"):
                logger.debug(" ** Build training graph: output")
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

        logger.debug(" ** Build training graph: setting class attributes")
        self.fetch_fn = fetch_fn
        self.model_vars = model_vars
        self.batch_model = batch_model

        self.learning_rate = learning_rate
        self.loss = batch_loss

        self.trainer_batch = trainer_batch
        self.trainer_full = trainer_full
        self.global_step = global_step

        self.gradient = batch_gradient
        # self.gradient_a = batch_gradient_a
        # self.gradient_b = batch_gradient_b

        self.train_op = trainer_full.train_op_GD

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
                summary_full_grad = tf.where(tf.is_nan(full_gradient), tf.zeros_like(full_gradient), full_gradient,
                                             name="full_gradient")
                # TODO: adjust this if gradient is changed
                tf.summary.histogram('batch_gradient', batch_trainers.gradient_by_variable(model_vars.params))
                tf.summary.histogram("full_gradient", summary_full_grad)
                tf.summary.scalar("full_gradient_median", tf.contrib.distributions.percentile(full_gradient, 50.))
                tf.summary.scalar("full_gradient_mean", tf.reduce_mean(full_gradient))

        self.saver = tf.train.Saver()
        self.merged_summary = tf.summary.merge_all()
