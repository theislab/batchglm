import abc
import logging
from typing import Union

import tensorflow as tf

import numpy as np

try:
    import anndata
except ImportError:
    anndata = None

from .external import train_utils, op_utils

logger = logging.getLogger(__name__)


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
            gradients_full = self.gradients_full_byfeature
            gradients_batch = self.gradients_batched_byfeature
        elif termination_type == "global":
            logger.debug(" ** Build gradients for training graph: global")
            gradients_full = self.gradients_full_global
            gradients_batch = self.gradients_batched_global
        else:
            raise ValueError("convergence_type %s not recognized." % termination_type)

        # Pad gradients to receive update tensors that match
        # the shape of model_vars.params.
        if train_mu and not train_r:
            if train_r:
                pass
            else:
                gradients_batch = tf.concat([
                    gradients_batch,
                    tf.zeros_like(self.model_vars.b)
                ], axis=0)
                gradients_full = tf.concat([
                    gradients_full,
                    tf.zeros_like(self.model_vars.b)
                ], axis=0)
        elif train_r:
            gradients_batch = tf.concat([
                tf.zeros_like(self.model_vars.a),
                gradients_batch
            ], axis=0)
            gradients_full = tf.concat([
                tf.zeros_like(self.model_vars.a),
                gradients_full
            ], axis=0)
        else:
            raise ValueError("No training necessary")

        self.gradients_full = gradients_full
        self.gradients_batch = gradients_batch

    @abc.abstractmethod
    def model_vars(self):
        pass

    @abc.abstractmethod
    def full_data_model(self):
        pass

    @abc.abstractmethod
    def batched_data_model(self):
        pass

    @abc.abstractmethod
    def batch_jac(self):
        pass

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

        return gradients_full

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

        return gradients_batch

    def gradients_full_global(self):
        gradients_full = tf.transpose(self.full_data_model.neg_jac_train)
        return gradients_full

    def gradients_batched_global(self):
        gradients_batch = tf.transpose(self.batch_jac.neg_jac)
        return gradients_batch


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
    nr_update_full: Union[tf.Tensor, None]
    nr_update_batched: Union[tf.Tensor, None]

    def __init__(
            self,
            termination_type,
            provide_optimizers,
            train_mu,
            train_r
    ):
        if termination_type == "by_feature":
            if provide_optimizers["nr"]:
                logger.debug(" ** Build newton training graph: by_feature, train mu and r")
                nr_update_full = self.nr_update_full_byfeature
                nr_update_batched = self.nr_update_batched_byfeature
            else:
                nr_update_full = None
                nr_update_batched = None
        elif termination_type == "global":
            if provide_optimizers["nr"]:
                logger.debug(" ** Build newton training graph: global, train mu and r")
                nr_update_full = self.nr_update_full_byfeature
                nr_update_batched = self.nr_update_batched_byfeature
            else:
                nr_update_full = None
                nr_update_batched = None
        else:
            raise ValueError("convergence_type %s not recognized." % termination_type)

        # Pad update vectors to receive update tensors that match
        # the shape of model_vars.params.
        if train_mu:
            if train_r:
                if provide_optimizers["nr"]:
                    pass
            else:
                if provide_optimizers["nr"]:
                    nr_update_full = tf.concat([
                        nr_update_full,
                        tf.zeros_like(self.model_vars.b)
                    ], axis=0)
                    nr_update_batched = tf.concat([
                        nr_update_batched,
                        tf.zeros_like(self.model_vars.b)
                    ], axis=0)
        elif train_r:
            if provide_optimizers["nr"]:
                nr_update_full = tf.concat([
                    tf.zeros_like(self.model_vars.a),
                    nr_update_full
                ], axis=0)
                nr_update_batched = tf.concat([
                    tf.zeros_like(self.model_vars.a),
                    nr_update_batched
                ], axis=0)
        else:
            raise ValueError("No training necessary")

        self.nr_update_full = nr_update_full
        self.nr_update_batched = nr_update_batched

    @abc.abstractmethod
    def model_vars(self):
        pass

    @abc.abstractmethod
    def full_data_model(self):
        pass

    @abc.abstractmethod
    def batched_data_model(self):
        pass

    @abc.abstractmethod
    def batch_jac(self):
        pass

    @abc.abstractmethod
    def batch_hessians(self):
        pass

    @property
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
            if not model_vars.converged[i]
            else tf.zeros([1, param_grad_vec.shape[1]])
            for i in range(self.model_vars.n_features)
        ], axis=0)
        nr_update_full = tf.transpose(delta_t_bygene_full)

        return nr_update_full

    @property
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

        return nr_update_batched

    @property
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

        return nr_update_full

    @property
    def nr_update_batched_global(self):
        # Batched data model:
        param_grad_vec_batched = self.batch_jac.neg_jac
        delta_batched_t = tf.squeeze(tf.matrix_solve_ls(
            self.batch_hessians.neg_hessian,
            tf.expand_dims(param_grad_vec_batched, axis=-1),
            fast=False
        ), axis=-1)
        nr_update_batched = tf.transpose(delta_batched_t)

        return nr_update_batched

class TrainerGraph:

    def __init__(
            self,
            feature_isnonzero
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
        assert (self.a.shape == (num_design_loc_params, num_features))
        assert (self.b.shape == (num_design_scale_params, num_features))

    @abc.abstractmethod
    def model_vars(self):
        pass

    @abc.abstractmethod
    def full_data_model(self):
        pass

    @abc.abstractmethod
    def batched_data_model(self):
        pass

    @abc.abstractmethod
    def gradients_full(self):
        pass

    @abc.abstractmethod
    def gradients_batch(self):
        pass

    @abc.abstractmethod
    def nr_update_full(self):
        pass

    @abc.abstractmethod
    def nr_update_batched(self):
        pass
