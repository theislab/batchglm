import logging
from typing import Union

import tensorflow as tf

import numpy as np

try:
    import anndata
except ImportError:
    anndata = None

from .model import _ProcessModel, _ModelVars, BasicModelGraph_GLM
from .external import TFEstimatorGraph
from .external import train_utils

logger = logging.getLogger(__name__)



class FullDataModelGraph_GLM:
    """
    Computational graph to evaluate model on full data set.

    Here, we assume that the model cannot be executed on the full data set
    for memory reasons and therefore divide the data set into batches,
    execute the model on these batches and summarise the resulting metrics
    across batches. FullDataModelGraph is therefore an extension of
    BasicModelGraph that distributes operations across batches of observations.

    The distribution is performed by the function map_model().
    The model metrics which can be collected are:

        - The model likelihood (cost function value).
        - Model Jacobian matrix for trainer parameters (for training).
        - Model Jacobian matrix for all parameters (for downstream usage,
        e.g. hypothesis tests which can also be performed on closed form MLEs).
        - Model Hessian matrix for trainer parameters (for training).
        - Model Hessian matrix for all parameters (for downstream usage,
        e.g. hypothesis tests which can also be performed on closed form MLEs).
    """
    log_likelihood: tf.Tensor
    norm_log_likelihood: tf.Tensor
    norm_neg_log_likelihood: tf.Tensor
    loss: tf.Tensor

    jac: tf.Tensor
    neg_jac: tf.Tensor
    hessian: tf.Tensor
    neg_hessian: tf.Tensor
    neg_jac_train: tf.Tensor
    neg_hessian_train: tf.Tensor

    noise_model: str

    def __init__(self):
        pass


class GradientGraph:
    """

    Define newton-rhapson updates and gradients depending on termination type
    and depending on whether data is batched.
    The formed have to be distinguished because gradients and parameter updates
    are set to zero (or not computed in the case of newton-raphson) for
    converged features if feature-wise termination is chosen.
    The latter have to be distinguished as there are different jacobians
    and hessians for the full and the batched data.
    """
    model_vars: tf.Tensor
    full_data_model: tf.Tensor
    batched_data_model: tf.Tensor
    batch_jac: tf.Tensor

    nr_update_full: Union[tf.Tensor, None]
    nr_update_batched: Union[tf.Tensor, None]

    def __init__(
            self,
            termination_type,
            train_mu,
            train_r
    ):
        if termination_type == "by_feature":
            logger.debug(" ** Build gradients for training graph: by_feature")
            self.gradients_full_byfeature()
            self.gradients_batched_byfeature()
        elif termination_type == "global":
            logger.debug(" ** Build gradients for training graph: global")
            self.gradients_full_global()
            self.gradients_batched_global()
        else:
            raise ValueError("convergence_type %s not recognized." % termination_type)

        # Pad gradients to receive update tensors that match
        # the shape of model_vars.params.
        if train_mu:
            if train_r:
                gradients_batch = self.gradients_batch_raw
                gradients_full = self.gradients_full_raw
            else:
                gradients_batch = tf.concat([
                    self.gradients_batch_raw,
                    tf.zeros_like(self.model_vars.b)
                ], axis=0)
                gradients_full = tf.concat([
                    self.gradients_full_raw,
                    tf.zeros_like(self.model_vars.b)
                ], axis=0)
        elif train_r:
            gradients_batch = tf.concat([
                tf.zeros_like(self.model_vars.a),
                self.gradients_batch_raw
            ], axis=0)
            gradients_full = tf.concat([
                tf.zeros_like(self.model_vars.a),
                self.gradients_full_raw
            ], axis=0)
        else:
            raise ValueError("No training necessary")

        self.gradients_full = gradients_full
        self.gradients_batch = gradients_batch

    def gradients_full_byfeature(self):
        gradients_full_all = tf.transpose(self.full_data_model.neg_jac_train)
        gradients_full = tf.concat([
            # tf.gradients(full_data_model.norm_neg_log_likelihood,
            #             model_vars.params_by_gene[i])[0]
            tf.expand_dims(gradients_full_all[:, i], axis=-1)
            if not self.model_vars.converged[i]
            else tf.zeros([param_grad_vec.shape[1], 1], dtype=dtype)
            for i in range(self.model_vars.n_features)
        ], axis=1)

        self.gradients_full_raw = gradients_full

    def gradients_batched_byfeature(self):
        gradients_batch_all = tf.transpose(self.batch_jac.neg_jac)
        gradients_batch = tf.concat([
            # tf.gradients(batch_model.norm_neg_log_likelihood,
            #             model_vars.params_by_gene[i])[0]
            tf.expand_dims(gradients_batch_all[:, i], axis=-1)
            if not self.model_vars.converged[i]
            else tf.zeros([param_grad_vec.shape[1], 1], dtype=dtype)
            for i in range(self.model_vars.n_features)
        ], axis=1)

        self.gradients_batch_raw = gradients_batch

    def gradients_full_global(self):
        gradients_full = tf.transpose(self.full_data_model.neg_jac_train)
        self.gradients_full_raw = gradients_full

    def gradients_batched_global(self):
        gradients_batch = tf.transpose(self.batch_jac.neg_jac)
        self.gradients_batch_raw = gradients_batch


class NewtonGraph:
    """

    Define newton-rhapson updates and gradients depending on termination type
    and depending on whether data is batched.
    The formed have to be distinguished because gradients and parameter updates
    are set to zero (or not computed in the case of newton-raphson) for
    converged features if feature-wise termination is chosen.
    The latter have to be distinguished as there are different jacobians
    and hessians for the full and the batched data.
    """
    model_vars: tf.Tensor
    full_data_model: tf.Tensor
    batched_data_model: tf.Tensor
    batch_jac: tf.Tensor
    batch_hessians: tf.Tensor

    nr_update_full: Union[tf.Tensor, None]
    nr_update_batched: Union[tf.Tensor, None]

    def __init__(
            self,
            termination_type,
            provide_optimizers,
            train_mu,
            train_r
    ):
        if provide_optimizers["nr"]:
            if termination_type == "by_feature":
                logger.debug(" ** Build newton training graph: by_feature, train mu and r")
                self.nr_update_full_byfeature()
                self.nr_update_batched_byfeature()
            elif termination_type == "global":
                logger.debug(" ** Build newton training graph: global, train mu and r")
                self.nr_update_full_global()
                self.nr_update_batched_global()
            else:
                raise ValueError("convergence_type %s not recognized." % termination_type)

            # Pad update vectors to receive update tensors that match
            # the shape of model_vars.params.
            if train_mu:
                if train_r:
                    nr_update_full = self.nr_update_full_raw
                    nr_update_batched = self.nr_update_batched_raw
                else:
                    nr_update_full = tf.concat([
                        self.nr_update_full_raw,
                        tf.zeros_like(self.model_vars.b)
                    ], axis=0)
                    nr_update_batched = tf.concat([
                        self.nr_update_batched_raw,
                        tf.zeros_like(self.model_vars.b)
                    ], axis=0)
            elif train_r:
                nr_update_full = tf.concat([
                    tf.zeros_like(self.model_vars.a),
                    self.nr_update_full_raw
                ], axis=0)
                nr_update_batched = tf.concat([
                    tf.zeros_like(self.model_vars.a),
                    self.nr_update_batched_raw
                ], axis=0)
            else:
                raise ValueError("No training necessary")
        else:
            nr_update_full = None
            nr_update_batched = None

        self.nr_update_full = nr_update_full
        self.nr_update_batched = nr_update_batched

    def nr_update_full_byfeature(self):
        # Full data model by gene:
        param_grad_vec = self.full_data_model.neg_jac_train
        # Compute parameter update for non-converged gene only.
        delta_t_bygene_nonconverged = tf.squeeze(tf.matrix_solve_ls(
            tf.gather(
                self.full_data_model.neg_hessian_train,
                indices=self.idx_nonconverged,
                axis=0),
            tf.expand_dims(
                tf.gather(
                    param_grad_vec,
                    indices=self.idx_nonconverged,
                    axis=0)
                , axis=-1),
            fast=False
        ), axis=-1)
        # Write parameter updates into matrix of size of all parameters which
        # contains zero entries for updates of already converged genes.
        delta_t_bygene_full = tf.concat([
            tf.gather(delta_t_bygene_nonconverged,
                      indices=np.where(self.idx_nonconverged == i)[0],
                      axis=0)
            if not self.model_vars.converged[i]
            else tf.zeros([1, param_grad_vec.shape[1]])
            for i in range(self.model_vars.n_features)
        ], axis=0)
        nr_update_full = tf.transpose(delta_t_bygene_full)

        self.nr_update_full_raw = nr_update_full
        return

    def nr_update_batched_byfeature(self):
        # Batched data model by gene:
        param_grad_vec_batched = self.batch_jac.neg_jac
        # Compute parameter update for non-converged gene only.
        delta_batched_t_bygene_nonconverged = tf.squeeze(tf.matrix_solve_ls(
            tf.gather(
                self.batch_hessians.neg_hessian,
                indices=self.idx_nonconverged,
                axis=0),
            tf.expand_dims(
                tf.gather(
                    param_grad_vec_batched,
                    indices=self.idx_nonconverged,
                    axis=0)
                , axis=-1),
            fast=False
        ), axis=-1)
        # Write parameter updates into matrix of size of all parameters which
        # contains zero entries for updates of already converged genes.
        delta_batched_t_bygene_full = tf.concat([
            tf.gather(delta_batched_t_bygene_nonconverged,
                      indices=np.where(self.idx_nonconverged == i)[0],
                      axis=0)
            if not self.model_vars.converged[i]
            else tf.zeros([1, param_grad_vec.shape[1]])
            for i in range(self.model_vars.n_features)
        ], axis=0)
        nr_update_batched = tf.transpose(delta_batched_t_bygene_full)

        self.nr_update_batched_raw = nr_update_batched
        return

    def nr_update_full_global(self):
        # Full data model:
        param_grad_vec = self.full_data_model.neg_jac_train
        delta_t = tf.squeeze(tf.matrix_solve_ls(
            self.full_data_model.neg_hessian_train,
            # (full_data_model.hessians + tf.transpose(full_data_model.hessians, perm=[0, 2, 1])) / 2,  # symmetrization, don't need this with closed forms
            tf.expand_dims(param_grad_vec, axis=-1),
            fast=False
        ), axis=-1)
        nr_update_full = tf.transpose(delta_t)

        self.nr_update_full_raw = nr_update_full
        return

    def nr_update_batched_global(self):
        # Batched data model:
        param_grad_vec_batched = self.batch_jac.neg_jac
        delta_batched_t = tf.squeeze(tf.matrix_solve_ls(
            self.batch_hessians.neg_hessian,
            tf.expand_dims(param_grad_vec_batched, axis=-1),
            fast=False
        ), axis=-1)
        nr_update_batched = tf.transpose(delta_batched_t)

        self.nr_update_batched_raw = nr_update_batched
        return

class _TrainerGraph:
    """

    """
    model_vars: _ModelVars
    full_data_model: FullDataModelGraph_GLM
    batched_data_model: BasicModelGraph_GLM
    gradients_full: tf.Tensor
    gradients_batch: tf.Tensor
    nr_update_full: tf.Tensor
    nr_update_batched: tf.Tensor

    def __init__(
            self,
            feature_isnonzero,
            provide_optimizers,
            dtype
    ):
        with tf.name_scope("training_graphs"):
            logger.debug(" ** Build training graphs")
            global_step = tf.train.get_or_create_global_step()

            # Create trainers that produce training operations.
            trainer_batch = train_utils.MultiTrainer(
                variables=[self.model_vars.params],
                gradients=[(self.gradients_batch, self.model_vars.params)],
                newton_delta=self.nr_update_batched,
                learning_rate=self.learning_rate,
                global_step=global_step,
                apply_gradients=lambda grad: tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad),
                provide_optimizers=provide_optimizers,
                name="batch_data_trainers"
            )

            trainer_full = train_utils.MultiTrainer(
                variables=[self.model_vars.params],
                gradients=[(self.gradients_full, self.model_vars.params)],
                newton_delta=self.nr_update_full,
                learning_rate=self.learning_rate,
                global_step=global_step,
                apply_gradients=lambda grad: tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad),
                provide_optimizers=provide_optimizers,
                name="full_data_trainers"
            )

        # Set up model gradient computation:
        with tf.name_scope("batch_gradient"):
            logger.debug(" ** Build training graph: batched data model gradients")
            batch_gradient = trainer_batch.plain_gradient_by_variable(self.model_vars.params)
            batch_gradient = tf.reduce_sum(tf.abs(batch_gradient), axis=0)

            # batch_gradient = tf.add_n(
            #     [tf.reduce_sum(tf.abs(grad), axis=0) for (grad, var) in batch_trainers.gradient])

        with tf.name_scope("full_gradient"):
            logger.debug(" ** Build training graph: full data model gradients")
            # use same gradient as the optimizers
            full_gradient = trainer_full.plain_gradient_by_variable(self.model_vars.params)
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
            bounds_min, bounds_max = self.param_bounds(dtype)

            param_nonzero_a = tf.broadcast_to(feature_isnonzero, [self.num_design_loc_params, self.num_features])
            alt_a = tf.concat([
                # intercept
                tf.broadcast_to(bounds_min["a"], [1, self.num_features]),
                # slope
                tf.zeros(shape=[self.num_design_loc_params - 1, self.num_features], dtype=self.model_vars.a.dtype),
            ], axis=0, name="alt_a")
            a = tf.where(
                param_nonzero_a,
                self.model_vars.a,
                alt_a,
                name="a"
            )

            param_nonzero_b = tf.broadcast_to(feature_isnonzero, [self.num_design_scale_params, self.num_features])
            alt_b = tf.concat([
                # intercept
                tf.broadcast_to(bounds_max["b"], [1, self.num_features]),
                # slope
                tf.zeros(shape=[self.num_design_scale_params - 1, self.num_features], dtype=self.model_vars.b.dtype),
            ], axis=0, name="alt_b")
            b = tf.where(
                param_nonzero_b,
                self.model_vars.b,
                alt_b,
                name="b"
            )

        self.trainer_batch = trainer_batch
        self.trainer_full = trainer_full
        self.global_step = global_step

        self.gradient = batch_gradient

        self.train_op = trainer_full.train_op_GD
        self.init_ops = []
        self.init_op = init_op

        # # ### set up class attributes
        self.a = a
        self.b = b
        assert (self.a.shape == (self.num_design_loc_params, self.num_features))
        assert (self.b.shape == (self.num_design_scale_params, self.num_features))


class EstimatorGraph_GLM(TFEstimatorGraph, GradientGraph, NewtonGraph):
    """
    The estimator graphs are all graph necessary to perform parameter updates and to
    summarise a current parameter estimate.

    The estimator graph class is divided into the following major sub graphs:

        - The input pipeline: Feed data for parameter updates.
        -
    """
    X: tf.Tensor

    a: tf.Tensor
    b: tf.Tensor

    noise_model: str

    def __init__(
            self,
            num_observations,
            num_features,
            num_design_loc_params,
            num_design_scale_params,
            graph: tf.Graph = None,
            batch_size: int = None,
    ):
        """

        :param num_observations: int
            Number of observations.
        :param num_features: int
            Number of features.
        :param num_design_loc_params: int
            Number of parameters per feature in mean model.
        :param num_design_scale_params: int
            Number of parameters per feature in scale model.
        :param graph: tf.Graph
        """
        TFEstimatorGraph.__init__(
            self=self,
            graph=graph
        )

        self.num_observations = num_observations
        self.num_features = num_features
        self.num_design_loc_params = num_design_loc_params
        self.num_design_scale_params = num_design_scale_params
        self.batch_size = batch_size

