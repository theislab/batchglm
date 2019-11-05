import abc
import logging
from typing import Union

import numpy as np
import tensorflow as tf

try:
    import anndata
except ImportError:
    anndata = None

from .model import ModelVarsGLM
from .fim import FIMGLM
from .hessians import HessiansGLM
from .jacobians import JacobiansGLM
from .external import TFEstimatorGraph
from .external import train_utils
from .external import pkg_constants

logger = logging.getLogger(__name__)


class FullDataModelGraphGLM:
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

    jac: JacobiansGLM
    neg_jac_train: tf.Tensor

    hessians: HessiansGLM
    neg_hessians_train: tf.Tensor

    fim: FIMGLM
    fim_train: tf.Tensor

    noise_model: str


class BatchedDataModelGraphGLM:
    """
    Computational graph to evaluate model on batches of data set.

    The model metrics of a batch which can be collected are:

        - The model likelihood (cost function value).
        - Model Jacobian matrix for trained parameters (for training).
        - Model Hessian matrix for trained parameters (for training).
        - Model Fisher information matrix for trained parameters (for training).
    """
    log_likelihood: tf.Tensor
    norm_log_likelihood: tf.Tensor
    norm_neg_log_likelihood: tf.Tensor
    loss: tf.Tensor

    neg_jac_train: tf.Tensor
    neg_hessians_train: tf.Tensor
    fim_train: tf.Tensor

    noise_model: str


class GradientGraphGLM:
    """

    Define newton-rhapson updates and gradients depending on whether data is batched.
    The has to be distinguished as there are different jacobians
    and hessians for the full and the batched data.
    """
    model_vars: ModelVarsGLM
    full_data_model: FullDataModelGraphGLM
    batched_data_model: BatchedDataModelGraphGLM

    def __init__(
            self,
            model_vars: ModelVarsGLM,
            full_data_model: FullDataModelGraphGLM,
            batched_data_model: BatchedDataModelGraphGLM,
            train_loc,
            train_scale
    ):
        self.gradients_full_raw = None
        self.gradients_batch_raw = None
        self.model_vars = model_vars
        self.full_data_model = full_data_model
        self.batched_data_model = batched_data_model

        if train_loc or train_scale:
            self.gradients_full()
            if self.batched_data_model is not None:
                self.gradients_batched()

            # Pad gradients to receive update tensors that match
            # the shape of model_vars.params.
            if train_loc:
                if train_scale:
                    if self.batched_data_model is not None:
                        gradients_batch = self.gradients_batch_raw
                    gradients_full = self.gradients_full_raw
                else:
                    if self.batched_data_model is not None:
                        gradients_batch = tf.concat([
                            self.gradients_batch_raw,
                            tf.zeros_like(self.model_vars.b_var)
                        ], axis=0)
                    gradients_full = tf.concat([
                        self.gradients_full_raw,
                        tf.zeros_like(self.model_vars.b_var)
                    ], axis=0)
            else:
                if self.batched_data_model is not None:
                    gradients_batch = tf.concat([
                        tf.zeros_like(self.model_vars.a_var),
                        self.gradients_batch_raw
                    ], axis=0)
                gradients_full = tf.concat([
                    tf.zeros_like(self.model_vars.a_var),
                    self.gradients_full_raw
                ], axis=0)
        else:
            # These gradients are returned for convergence evaluation.
            # In this case, closed form estimates were used, one could
            # still evaluate the gradients here but we do not do
            # this to speed up run time.
            if self.batched_data_model is not None:
                gradients_batch = tf.zeros_like(self.model_vars.params)
            gradients_full = tf.zeros_like(self.model_vars.params)

        # Save attributes necessary for reinitialization:
        self.train_loc = train_loc
        self.train_scale = train_scale

        self.gradients_full = gradients_full
        if self.batched_data_model is not None:
            self.gradients_batch = gradients_batch
        else:
            self.gradients_batch = None

    def gradients_full(self):
        gradients_full = tf.transpose(self.full_data_model.neg_jac_train)
        self.gradients_full_raw = gradients_full

    def gradients_batched(self):
        gradients_batch = tf.transpose(self.batched_data_model.neg_jac_train)
        self.gradients_batch_raw = gradients_batch


class NewtonGraphGLM:
    """
    Define update vectors which require a matrix inversion: Newton-Raphson and
    IRLS updates.

    Define newton-type updates and gradients depending on whether data is batched.
    This has to be distinguished as there are different jacobians
    and hessians for the full and the batched data.
    """
    model_vars: tf.Tensor
    full_data_model: FullDataModelGraphGLM
    batched_data_model: BatchedDataModelGraphGLM

    nr_update_full: Union[tf.Tensor, None]
    nr_update_batched: Union[tf.Tensor, None]
    nr_tr_update_full: Union[tf.Tensor, None]
    nr_tr_update_batched: Union[tf.Tensor, None]

    irls_update_full: Union[tf.Tensor, None]
    irls_update_batched: Union[tf.Tensor, None]
    irls_tr_update_full: Union[tf.Tensor, None]
    irls_tr_update_batched: Union[tf.Tensor, None]

    nr_tr_radius: Union[tf.Variable, None]
    nr_tr_pred_cost_gain_full: Union[tf.Tensor, None]
    nr_tr_pred_cost_gain_batched: Union[tf.Tensor, None]

    irls_tr_radius: Union[tf.Variable, None]
    irls_tr_pred_cost_gain_full: Union[tf.Tensor, None]
    irls_tr_pred_cost_gain_batched: Union[tf.Tensor, None]

    def __init__(
            self,
            provide_optimizers,
            train_mu,
            train_r,
            dtype
    ):
        if train_mu or train_r:
            if provide_optimizers["nr"] or provide_optimizers["nr_tr"]:
                if self.batched_data_model is None:
                    batched_lhs = None
                    batched_rhs = None
                else:
                    batched_lhs = self.batched_data_model.neg_hessians_train
                    batched_rhs = self.batched_data_model.neg_jac_train

                nr_update_full_raw, nr_update_batched_raw = self.build_updates_nr(
                    full_lhs=self.full_data_model.neg_hessians_train,
                    batched_lhs=batched_lhs,
                    full_rhs=self.full_data_model.neg_jac_train,
                    batched_rhs=batched_rhs,
                    psd=False
                )
                nr_update_full, nr_update_batched = self.pad_updates(
                    train_mu=train_mu,
                    train_r=train_r,
                    update_full_raw=nr_update_full_raw,
                    update_batched_raw=nr_update_batched_raw
                )

                self.nr_tr_x_step_full = tf.Variable(tf.zeros_like(nr_update_full))
                if self.batched_data_model is None:
                    self.nr_tr_x_step_batched = None
                else:
                    self.nr_tr_x_step_batched = tf.Variable(tf.zeros_like(nr_update_batched))
            else:
                nr_update_full = None
                nr_update_batched = None

            if provide_optimizers["nr_tr"]:
                self.nr_tr_radius = tf.Variable(
                    np.zeros(shape=[self.model_vars.n_features]) + pkg_constants.TRUST_REGION_RADIUS_INIT,
                    dtype=dtype
                )
                self.nr_tr_ll_prev_full = tf.Variable(np.zeros(shape=[self.model_vars.n_features]))
                self.nr_tr_pred_gain_full = tf.Variable(np.zeros(shape=[self.model_vars.n_features]))

                if self.batched_data_model is None:
                    self.nr_tr_ll_prev_batched = None
                    self.nr_tr_pred_gain_batched = None
                else:
                    self.nr_tr_ll_prev_batched = tf.Variable(np.zeros(shape=[self.model_vars.n_features]))
                    self.nr_tr_pred_gain_batched = tf.Variable(np.zeros(shape=[self.model_vars.n_features]))

                n_obs = tf.cast(self.full_data_model.num_observations, dtype=dtype)

                nr_tr_proposed_vector_full = self.trust_region_newton_update(
                    update_raw=nr_update_full_raw,
                    radius_container=self.nr_tr_radius,
                    n_obs=self.num_observations_tf
                )
                nr_tr_pred_cost_gain_full = self.trust_region_newton_cost_gain(
                    proposed_vector=nr_tr_proposed_vector_full,
                    neg_jac=self.full_data_model.neg_jac_train,
                    hessian_fim=self.full_data_model.neg_hessians_train,
                    n_obs=self.num_observations_tf
                )

                if self.batched_data_model is not None:
                    nr_tr_proposed_vector_batched = self.trust_region_newton_update(
                        update_raw=nr_update_batched_raw,
                        radius_container=self.nr_tr_radius,
                        n_obs=self.batch_size_tf
                    )
                    nr_tr_pred_cost_gain_batched = self.trust_region_newton_cost_gain(
                        proposed_vector=nr_tr_proposed_vector_full,
                        neg_jac=self.batched_data_model.neg_jac_train,
                        hessian_fim=self.batched_data_model.neg_hessians_train,
                        n_obs=self.batch_size_tf
                    )
                else:
                    nr_tr_pred_cost_gain_batched = None
                    nr_tr_proposed_vector_batched = None

                nr_tr_proposed_vector_full_pad, nr_tr_proposed_vector_batched_pad = self.pad_updates(
                    train_mu=train_mu,
                    train_r=train_r,
                    update_full_raw=nr_tr_proposed_vector_full,
                    update_batched_raw=nr_tr_proposed_vector_batched
                )

                train_ops_nr_tr_full = self.trust_region_ops(
                    likelihood_container=self.nr_tr_ll_prev_full,
                    proposed_vector=nr_tr_proposed_vector_full_pad,
                    proposed_vector_container=self.nr_tr_x_step_full,
                    proposed_gain=nr_tr_pred_cost_gain_full,
                    proposed_gain_container=self.nr_tr_pred_gain_full,
                    radius_container=self.nr_tr_radius,
                    dtype=dtype
                )
                if self.batched_data_model is not None:
                    train_ops_nr_tr_batched = self.trust_region_ops(
                        likelihood_container=self.nr_tr_ll_prev_batched,
                        proposed_vector=nr_tr_proposed_vector_batched_pad,
                        proposed_vector_container=self.nr_tr_x_step_batched,
                        proposed_gain=nr_tr_pred_cost_gain_batched,
                        proposed_gain_container=self.nr_tr_pred_gain_batched,
                        radius_container=self.nr_tr_radius,
                        dtype=dtype
                    )
                else:
                    train_ops_nr_tr_batched = None
            else:
                train_ops_nr_tr_full = None
                train_ops_nr_tr_batched = None
                self.nr_tr_radius = tf.Variable(np.array([np.inf]), dtype=dtype)

            if provide_optimizers["irls"] or provide_optimizers["irls_tr"] or \
                    provide_optimizers["irls_gd"] or provide_optimizers["irls_gd_tr"]:
                # Compute a and b model updates separately.
                if train_mu:
                    # The FIM of the mean model is guaranteed to be
                    # positive semi-definite and can therefore be inverted
                    # with the Cholesky decomposition. This information is
                    # passed here with psd=True.
                    if self.batched_data_model is None:
                        batched_lhs = None
                        batched_rhs = None
                    else:
                        batched_lhs = self.batched_data_model.fim_a
                        batched_rhs = self.batched_data_model.neg_jac_a

                    irls_update_a_full, irls_update_a_batched = self.build_updates_nr(
                        full_lhs=self.full_data_model.fim_a,
                        batched_lhs=batched_lhs,
                        full_rhs=self.full_data_model.neg_jac_a,
                        batched_rhs=batched_rhs,
                        psd=True
                    )
                else:
                    irls_update_a_full = None
                    irls_update_a_batched = None

                if train_r:
                    if self.batched_data_model is None:
                        batched_lhs = None
                        batched_rhs = None
                    else:
                        batched_lhs = self.batched_data_model.fim_b
                        batched_rhs = self.batched_data_model.neg_jac_b
                    if provide_optimizers["irls"] or provide_optimizers["irls_tr"]:
                        irls_update_b_full, irls_update_b_batched = self.build_updates_nr(
                            full_lhs=self.full_data_model.fim_b,
                            batched_lhs=batched_lhs,
                            full_rhs=self.full_data_model.neg_jac_b,
                            batched_rhs=batched_rhs,
                            psd=False
                        )
                    else:
                        irls_update_b_full = None
                        irls_update_b_batched = None
                    if provide_optimizers["irls_gd"] or provide_optimizers["irls_gd_tr"]:
                        if self.batched_data_model is not None:
                            batched_jac = self.batched_data_model.neg_jac_b
                        else:
                            batched_jac = None
                        irls_gd_update_b_full, irls_gd_update_b_batched = self.build_updates_gd(
                            full_jac=self.full_data_model.neg_jac_b,
                            batched_jac=batched_jac,
                        )
                    else:
                        irls_gd_update_b_full = None
                        irls_gd_update_b_batched = None
                else:
                    irls_update_b_full = None
                    irls_update_b_batched = None
                    irls_gd_update_b_full = None
                    irls_gd_update_b_batched = None

            if provide_optimizers["irls"]:
                if train_mu and train_r:
                    irls_update_full_raw = tf.concat([irls_update_a_full, irls_update_b_full], axis=0)
                    if self.batched_data_model is not None:
                        irls_update_batched_raw = tf.concat([irls_update_a_batched, irls_update_b_batched], axis=0)
                    else:
                        irls_update_batched_raw = None
                elif train_mu:
                    irls_update_full_raw = irls_update_a_full
                    if self.batched_data_model is not None:
                        irls_update_batched_raw = irls_update_a_batched
                    else:
                        irls_update_batched_raw = None
                elif train_r:
                    irls_update_full_raw = irls_update_b_full
                    if self.batched_data_model is not None:
                        irls_update_batched_raw = irls_update_b_batched
                    else:
                        irls_update_batched_raw = None
                else:
                    irls_update_full_raw = None
                    if self.batched_data_model is not None:
                        irls_update_batched_raw = None
                    else:
                        irls_update_batched_raw = None

                irls_update_full, irls_update_batched = self.pad_updates(
                    train_mu=train_mu,
                    train_r=train_r,
                    update_full_raw=irls_update_full_raw,
                    update_batched_raw=irls_update_batched_raw
                )

                self.irls_tr_x_step_full = tf.Variable(tf.zeros_like(irls_update_full))
                if self.batched_data_model is None:
                    self.irls_tr_x_step_batched = None
                else:
                    self.irls_tr_x_step_batched = tf.Variable(tf.zeros_like(irls_update_full))
            else:
                irls_update_full = None
                irls_update_batched = None

            if provide_optimizers["irls_gd"]:
                if train_mu and train_r:
                    irls_gd_update_full_raw = tf.concat([irls_update_a_full, irls_gd_update_b_full], axis=0)
                    if self.batched_data_model is not None:
                        irls_gd_update_batched_raw = tf.concat([irls_update_a_batched, irls_gd_update_b_batched], axis=0)
                    else:
                        irls_gd_update_batched_raw = None
                elif train_mu:
                    irls_gd_update_full_raw = irls_update_a_full
                    if self.batched_data_model is not None:
                        irls_gd_update_batched_raw = irls_update_a_batched
                    else:
                        irls_gd_update_batched_raw = None
                elif train_r:
                    irls_gd_update_full_raw = irls_gd_update_b_full
                    if self.batched_data_model is not None:
                        irls_gd_update_batched_raw = irls_gd_update_b_batched
                    else:
                        irls_gd_update_batched_raw = None
                else:
                    irls_gd_update_full_raw = None
                    if self.batched_data_model is not None:
                        irls_gd_update_batched_raw = None
                    else:
                        irls_gd_update_batched_raw = None

                irls_gd_update_full, irls_gd_update_batched = self.pad_updates(
                    train_mu=train_mu,
                    train_r=train_r,
                    update_full_raw=irls_gd_update_full_raw,
                    update_batched_raw=irls_gd_update_batched_raw
                )

                self.irls_gd_tr_x_step_full = tf.Variable(tf.zeros_like(irls_gd_update_full))
                if self.batched_data_model is None:
                    self.irls_gd_tr_x_step_batched = None
                else:
                    self.irls_gd_tr_x_step_batched = tf.Variable(tf.zeros_like(irls_gd_update_batched))
            else:
                irls_gd_update_full = None
                irls_gd_update_batched = None

            if provide_optimizers["irls_tr"] or provide_optimizers["irls_gd_tr"]:
                self.irls_tr_radius = tf.Variable(
                    np.zeros(shape=[self.model_vars.n_features]) + pkg_constants.TRUST_REGION_RADIUS_INIT,
                    dtype=dtype
                )
                self.irls_tr_ll_prev_full = tf.Variable(np.zeros(shape=[self.model_vars.n_features]))
                self.irls_tr_pred_gain_full = tf.Variable(np.zeros(shape=[self.model_vars.n_features]))

                if self.batched_data_model is None:
                    self.irls_tr_ll_prev_batched = None
                    self.irls_tr_pred_gain_batched = None
                else:
                    self.irls_tr_ll_prev_batched = tf.Variable(np.zeros(shape=[self.model_vars.n_features]))
                    self.irls_tr_pred_gain_batched = tf.Variable(np.zeros(shape=[self.model_vars.n_features]))

                if train_mu:
                    irls_tr_proposed_vector_full_a = self.trust_region_newton_update(
                        update_raw=irls_update_a_full,
                        radius_container=self.irls_tr_radius,
                        n_obs=self.num_observations_tf
                    )
                    irls_tr_pred_cost_gain_full_a = self.trust_region_newton_cost_gain(
                        proposed_vector=irls_tr_proposed_vector_full_a,
                        neg_jac=self.full_data_model.neg_jac_a,
                        hessian_fim=self.full_data_model.fim_a,
                        n_obs=self.num_observations_tf
                    )
                else:
                    irls_tr_proposed_vector_full_a = None
                    irls_tr_pred_cost_gain_full_a = None

                if train_r:
                    if provide_optimizers["irls_tr"]:
                        irls_tr_proposed_vector_full_b = self.trust_region_newton_update(
                            update_raw=irls_update_b_full,
                            radius_container=self.irls_tr_radius,
                            n_obs=self.num_observations_tf
                        )
                        irls_tr_pred_cost_gain_full_b = self.trust_region_newton_cost_gain(
                            proposed_vector=irls_tr_proposed_vector_full_b,
                            neg_jac=self.full_data_model.neg_jac_b,
                            hessian_fim=self.full_data_model.fim_b,
                            n_obs=self.num_observations_tf
                        )
                    else:
                        irls_tr_proposed_vector_full_b = None
                        irls_tr_pred_cost_gain_full_b = None

                    if provide_optimizers["irls_gd_tr"]:
                        irls_gd_tr_proposed_vector_full_b = self.trust_region_linear_update(
                            update_raw=irls_gd_update_b_full,
                            radius_container=self.irls_tr_radius,
                            n_obs=self.num_observations_tf
                        )
                        irls_gd_tr_pred_cost_gain_full_b = self.trust_region_linear_cost_gain(
                            proposed_vector=irls_gd_tr_proposed_vector_full_b,
                            neg_jac=self.full_data_model.neg_jac_b,
                            n_obs=self.num_observations_tf
                        )
                    else:
                        irls_gd_tr_proposed_vector_full_b = None
                        irls_gd_tr_pred_cost_gain_full_b = None
                else:
                    irls_tr_proposed_vector_full_b = None
                    irls_tr_pred_cost_gain_full_b = None
                    irls_gd_tr_proposed_vector_full_b = None
                    irls_gd_tr_pred_cost_gain_full_b = None

                if self.batched_data_model is not None:
                    if train_mu:
                        irls_tr_proposed_vector_batched_a = self.trust_region_newton_update(
                            update_raw=irls_update_a_batched,
                            radius_container=self.irls_tr_radius,
                            n_obs=self.batch_size_tf
                        )
                        irls_tr_pred_cost_gain_batched_a = self.trust_region_newton_cost_gain(
                            proposed_vector=irls_tr_proposed_vector_batched_a,
                            neg_jac=self.batched_data_model.neg_jac_a,
                            hessian_fim=self.batched_data_model.fim_a,
                            n_obs=self.batch_size_tf
                        )
                    else:
                        irls_tr_proposed_vector_batched_a = None
                        irls_tr_pred_cost_gain_batched_a = None

                    if train_r:
                        if provide_optimizers["irls_tr"]:
                            irls_tr_proposed_vector_batched_b = self.trust_region_newton_update(
                                update_raw=irls_update_b_batched,
                                radius_container=self.irls_tr_radius,
                                n_obs=self.batch_size_tf
                            )
                            irls_tr_pred_cost_gain_batched_b = self.trust_region_newton_cost_gain(
                                proposed_vector=irls_tr_proposed_vector_batched_b,
                                neg_jac=self.batched_data_model.neg_jac_b,
                                hessian_fim=self.batched_data_model.fim_b,
                                n_obs=self.batch_size_tf
                            )
                        else:
                            irls_tr_proposed_vector_batched_b = None
                            irls_tr_pred_cost_gain_batched_b = None

                        if provide_optimizers["irls_gd_tr"]:
                            irls_gd_tr_proposed_vector_batched_b = self.trust_region_linear_update(
                                update_raw=irls_gd_update_b_batched,
                                radius_container=self.irls_tr_radius,
                                n_obs=self.batch_size_tf
                            )
                            irls_gd_tr_pred_cost_gain_batched_b = self.trust_region_linear_cost_gain(
                                proposed_vector=irls_gd_tr_proposed_vector_batched_b,
                                neg_jac=self.batched_data_model.neg_jac_b,
                                n_obs=self.batch_size_tf
                            )
                        else:
                            irls_gd_tr_proposed_vector_batched_b = None
                            irls_gd_tr_pred_cost_gain_batched_b = None
                    else:
                        irls_tr_proposed_vector_batched_b = None
                        irls_tr_pred_cost_gain_batched_b = None
                        irls_gd_tr_proposed_vector_batched_b = None
                        irls_gd_tr_pred_cost_gain_batched_b = None

                if train_mu and train_r:
                    if provide_optimizers["irls_tr"]:
                        irls_tr_update_full_raw = tf.concat([irls_tr_proposed_vector_full_a,
                                                             irls_tr_proposed_vector_full_b], axis=0)
                        irls_tr_pred_cost_gain_full = tf.add(irls_tr_pred_cost_gain_full_a,
                                                             irls_tr_pred_cost_gain_full_b)
                    else:
                        irls_tr_update_full_raw = None
                        irls_tr_pred_cost_gain_full = None

                    if provide_optimizers["irls_gd_tr"]:
                        irls_gd_tr_update_full_raw = tf.concat([irls_tr_proposed_vector_full_a,
                                                                irls_gd_tr_proposed_vector_full_b], axis=0)
                        irls_gd_tr_pred_cost_gain_full = tf.add(irls_tr_pred_cost_gain_full_a,
                                                                irls_gd_tr_pred_cost_gain_full_b)
                    else:
                        irls_gd_tr_update_full_raw = None
                        irls_gd_tr_pred_cost_gain_full = None

                    if self.batched_data_model is not None:
                        if provide_optimizers["irls_tr"]:
                            irls_tr_update_batched_raw = tf.concat([irls_tr_proposed_vector_batched_a,
                                                                    irls_tr_proposed_vector_batched_b], axis=0)
                            irls_tr_pred_cost_gain_batched = tf.add(irls_tr_pred_cost_gain_batched_a,
                                                                    irls_tr_pred_cost_gain_batched_b)
                        else:
                            irls_tr_update_batched_raw = None
                            irls_tr_pred_cost_gain_batched = None

                        if provide_optimizers["irls_gd_tr"]:
                            irls_gd_tr_update_batched_raw = tf.concat([irls_tr_proposed_vector_batched_a,
                                                                       irls_gd_tr_proposed_vector_batched_b], axis=0)
                            irls_gd_tr_pred_cost_gain_batched = tf.add(irls_tr_pred_cost_gain_batched_a,
                                                                       irls_gd_tr_pred_cost_gain_batched_b)
                        else:
                            irls_gd_tr_update_batched_raw = None
                            irls_gd_tr_pred_cost_gain_batched = None
                    else:
                        irls_tr_update_batched_raw = None
                        irls_gd_tr_update_batched_raw = None
                        irls_tr_pred_cost_gain_batched = None
                        irls_gd_tr_pred_cost_gain_batched = None
                elif train_mu and not train_r:
                    irls_tr_update_full_raw = irls_tr_proposed_vector_full_a
                    irls_gd_tr_update_full_raw = irls_tr_proposed_vector_full_a
                    irls_tr_pred_cost_gain_full = irls_tr_pred_cost_gain_full_a
                    irls_gd_tr_pred_cost_gain_full = irls_tr_pred_cost_gain_full_a
                    if self.batched_data_model is not None:
                        irls_tr_update_batched_raw = irls_tr_proposed_vector_batched_a
                        irls_gd_tr_update_batched_raw = irls_tr_proposed_vector_batched_a
                        irls_tr_pred_cost_gain_batched = irls_tr_pred_cost_gain_batched_a
                        irls_gd_tr_pred_cost_gain_batched = irls_tr_pred_cost_gain_batched_a
                    else:
                        irls_tr_update_batched_raw = None
                        irls_gd_tr_update_batched_raw = None
                        irls_tr_pred_cost_gain_batched = None
                        irls_gd_tr_pred_cost_gain_batched = None
                elif not train_mu and train_r:
                    if provide_optimizers["irls_tr"]:
                        irls_tr_update_full_raw = irls_tr_proposed_vector_full_b
                        irls_tr_pred_cost_gain_full = irls_tr_pred_cost_gain_full_b
                    else:
                        irls_tr_update_full_raw = None
                        irls_tr_pred_cost_gain_full = None

                    if provide_optimizers["irls_gd_tr"]:
                        irls_gd_tr_update_full_raw = irls_gd_tr_proposed_vector_full_b
                        irls_gd_tr_pred_cost_gain_full = irls_gd_tr_pred_cost_gain_full_b
                    else:
                        irls_gd_tr_update_full_raw = None
                        irls_gd_tr_pred_cost_gain_full = None

                    if self.batched_data_model is not None:
                        if provide_optimizers["irls_tr"]:
                            irls_tr_update_batched_raw = irls_tr_proposed_vector_batched_b
                            irls_tr_pred_cost_gain_batched = irls_tr_pred_cost_gain_batched_b
                        else:
                            irls_tr_update_batched_raw = None
                            irls_tr_pred_cost_gain_batched = None

                        if provide_optimizers["irls_gd_tr"]:
                            irls_gd_tr_update_batched_raw = irls_gd_tr_proposed_vector_batched_b
                            irls_gd_tr_pred_cost_gain_batched = irls_gd_tr_pred_cost_gain_batched_b
                        else:
                            irls_gd_tr_update_batched_raw = None
                            irls_gd_tr_pred_cost_gain_batched = None
                    else:
                        irls_tr_update_batched_raw = None
                        irls_gd_tr_update_batched_raw = None
                        irls_tr_pred_cost_gain_batched = None
                        irls_gd_tr_pred_cost_gain_batched = None
                else:
                    assert False

                if provide_optimizers["irls_tr"]:
                    irls_tr_update_full, irls_tr_update_batched = self.pad_updates(
                        train_mu=train_mu,
                        train_r=train_r,
                        update_full_raw=irls_tr_update_full_raw,
                        update_batched_raw=irls_tr_update_batched_raw
                    )
                else:
                    irls_tr_update_full = None
                    irls_tr_update_batched = None

                if provide_optimizers["irls_gd_tr"]:
                    irls_gd_tr_update_full, irls_gd_tr_update_batched = self.pad_updates(
                        train_mu=train_mu,
                        train_r=train_r,
                        update_full_raw=irls_gd_tr_update_full_raw,
                        update_batched_raw=irls_gd_tr_update_batched_raw
                    )
                else:
                    irls_gd_tr_update_full = None
                    irls_gd_tr_update_batched = None

                if provide_optimizers["irls_tr"] or provide_optimizers["irls_gd_tr"]:
                    self.irls_tr_x_step_full = tf.Variable(tf.zeros_like(self.model_vars.params))
                    if self.batched_data_model is None:
                        self.irls_tr_x_step_batched = None
                    else:
                        self.irls_tr_x_step_batched = tf.Variable(tf.zeros_like(self.model_vars.params))
                else:
                    self.irls_tr_x_step_full = None
                    self.irls_tr_x_step_batched = None

                if provide_optimizers["irls_tr"]:
                    train_ops_irls_tr_full = self.trust_region_ops(
                        likelihood_container=self.irls_tr_ll_prev_full,
                        proposed_vector=irls_tr_update_full,
                        proposed_vector_container=self.irls_tr_x_step_full,
                        proposed_gain=irls_tr_pred_cost_gain_full,
                        proposed_gain_container=self.irls_tr_pred_gain_full,
                        radius_container=self.irls_tr_radius,
                        dtype=dtype
                    )
                    if self.batched_data_model is not None:
                        train_ops_irls_tr_batched = self.trust_region_ops(
                            likelihood_container=self.irls_tr_ll_prev_batched,
                            proposed_vector=irls_tr_update_batched,
                            proposed_vector_container=self.irls_tr_x_step_batched,
                            proposed_gain=irls_tr_pred_cost_gain_batched,
                            proposed_gain_container=self.irls_tr_pred_gain_batched,
                            radius_container=self.irls_tr_radius,
                            dtype=dtype
                        )
                    else:
                        train_ops_irls_tr_batched = None
                else:
                    train_ops_irls_tr_full = None
                    train_ops_irls_tr_batched = None

                if provide_optimizers["irls_gd_tr"]:
                    train_ops_irls_gd_tr_full = self.trust_region_ops(
                        likelihood_container=self.irls_tr_ll_prev_full,
                        proposed_vector=irls_gd_tr_update_full,
                        proposed_vector_container=self.irls_tr_x_step_full,
                        proposed_gain=irls_gd_tr_pred_cost_gain_full,
                        proposed_gain_container=self.irls_tr_pred_gain_full,
                        radius_container=self.irls_tr_radius,
                        dtype=dtype
                    )
                    if self.batched_data_model is not None:
                        train_ops_irls_gd_tr_batched = self.trust_region_ops(
                            likelihood_container=self.irls_tr_ll_prev_batched,
                            proposed_vector=irls_gd_tr_update_batched,
                            proposed_vector_container=self.irls_tr_x_step_batched,
                            proposed_gain=irls_gd_tr_pred_cost_gain_batched,
                            proposed_gain_container=self.irls_tr_pred_gain_batched,
                            radius_container=self.irls_tr_radius,
                            dtype=dtype
                        )
                    else:
                        train_ops_irls_gd_tr_batched = None
                else:
                    self.irls_gd_tr_x_step_full = None
                    self.irls_gd_tr_x_step_batched = None
                    train_ops_irls_gd_tr_full = None
                    train_ops_irls_gd_tr_batched = None
            else:
                train_ops_irls_tr_full = None
                train_ops_irls_tr_batched = None
                train_ops_irls_gd_tr_full = None
                train_ops_irls_gd_tr_batched = None
                self.irls_tr_radius = tf.Variable(np.array([np.inf]), dtype=dtype)
        else:
            nr_update_full = None
            nr_update_batched = None
            train_ops_nr_tr_full = None
            train_ops_nr_tr_batched = None

            irls_update_full = None
            irls_update_batched = None
            irls_gd_update_full = None
            irls_gd_update_batched = None
            train_ops_irls_tr_full = None
            train_ops_irls_tr_batched = None
            train_ops_irls_gd_tr_full = None
            train_ops_irls_gd_tr_batched = None

            self.nr_tr_radius = tf.Variable(np.array([np.inf]), dtype=dtype)
            self.irls_tr_radius = tf.Variable(np.array([np.inf]), dtype=dtype)

        self.nr_update_full = nr_update_full
        self.nr_update_batched = nr_update_batched
        self.train_ops_nr_tr_full = train_ops_nr_tr_full
        self.train_ops_nr_tr_batched = train_ops_nr_tr_batched

        self.irls_update_full = irls_update_full
        self.irls_update_batched = irls_update_batched
        self.irls_gd_update_full = irls_gd_update_full
        self.irls_gd_update_batched = irls_gd_update_batched
        self.train_ops_irls_tr_full = train_ops_irls_tr_full
        self.train_ops_irls_tr_batched = train_ops_irls_tr_batched
        self.train_ops_irls_gd_tr_full = train_ops_irls_gd_tr_full
        self.train_ops_irls_gd_tr_batched = train_ops_irls_gd_tr_batched

    def build_updates_nr(
            self,
            full_lhs,
            batched_rhs,
            full_rhs,
            batched_lhs,
            psd
    ):
        update_full = self.newton_type_update(
            lhs=full_lhs,
            rhs=full_rhs,
            psd=psd
        )
        if batched_lhs is not None:
            update_batched = self.newton_type_update(
                lhs=batched_lhs,
                rhs=batched_rhs,
                psd=psd and pkg_constants.CHOLESKY_LSTSQS_BATCHED  # This can be unstable even for fim_a.
            )
        else:
            update_batched = None

        return update_full, update_batched

    def build_updates_gd(
            self,
            full_jac,
            batched_jac
    ):
        update_full = tf.transpose(full_jac)
        if batched_jac is not None:
            update_batched = tf.transpose(batched_jac)
        else:
            update_batched = None

        return update_full, update_batched

    def pad_updates(
            self,
            update_full_raw,
            update_batched_raw,
            train_mu,
            train_r
    ):
        # Pad update vectors to receive update tensors that match
        # the shape of model_vars.params.
        if train_mu:
            if train_r:
                netwon_type_update_full = update_full_raw
                newton_type_update_batched = update_batched_raw
            else:
                netwon_type_update_full = tf.concat([
                    update_full_raw,
                    tf.zeros_like(self.model_vars.b_var)
                ], axis=0)
                if update_batched_raw is not None:
                    newton_type_update_batched = tf.concat([
                        update_batched_raw,
                        tf.zeros_like(self.model_vars.b_var)
                    ], axis=0)
                else:
                    newton_type_update_batched = None
        elif train_r:
            netwon_type_update_full = tf.concat([
                tf.zeros_like(self.model_vars.a_var),
                update_full_raw
            ], axis=0)
            if update_batched_raw is not None:
                newton_type_update_batched = tf.concat([
                    tf.zeros_like(self.model_vars.a_var),
                    update_batched_raw
                ], axis=0)
            else:
                newton_type_update_batched = None
        else:
            raise ValueError("No training necessary")

        return netwon_type_update_full, newton_type_update_batched

    def newton_type_update(
            self,
            lhs,
            rhs,
            psd
    ):
        delta_t = tf.squeeze(tf.linalg.lstsq(
            lhs,
            tf.expand_dims(rhs, axis=-1),
            fast=psd and pkg_constants.CHOLESKY_LSTSQS
        ), axis=-1)
        update_tensor = tf.transpose(delta_t)

        return update_tensor

    def trust_region_newton_update(
            self,
            update_raw,
            radius_container,
            n_obs
    ):
        update_magnitude_sq = tf.reduce_sum(tf.square(update_raw), axis=0)
        update_magnitude = tf.where(
            condition=update_magnitude_sq > 0,
            x=tf.sqrt(update_magnitude_sq),
            y=tf.zeros_like(update_magnitude_sq)
        )
        update_magnitude_inv = tf.where(
            condition=update_magnitude > 0,
            x=tf.divide(
                tf.ones_like(update_magnitude),
                update_magnitude
            ),
            y=tf.zeros_like(update_magnitude)
        )
        update_norm = tf.multiply(update_raw,update_magnitude_inv)
        update_scale = tf.minimum(
            radius_container,
            update_magnitude
        )
        proposed_vector = tf.multiply(
            update_norm,
            update_scale
        )

        return proposed_vector

    def trust_region_linear_update(
            self,
            update_raw,
            radius_container,
            n_obs
    ):
        update_magnitude_sq = tf.reduce_sum(tf.square(update_raw), axis=0)
        update_magnitude = tf.where(
            condition=update_magnitude_sq > 0,
            x=tf.sqrt(update_magnitude_sq),
            y=tf.zeros_like(update_magnitude_sq)
        )
        update_magnitude_inv = tf.where(
            condition=update_magnitude > 0,
            x=tf.divide(
                tf.ones_like(update_magnitude),
                update_magnitude
            ),
            y=tf.zeros_like(update_magnitude)
        )
        update_norm = tf.multiply(update_raw,update_magnitude_inv)
        update_scale = tf.minimum(
            radius_container,
            update_magnitude / n_obs  # learning rate = 1
        )
        proposed_vector = tf.multiply(
            update_norm,
            update_scale
        )

        return proposed_vector

    def trust_region_newton_cost_gain(
            self,
            proposed_vector,
            neg_jac,
            hessian_fim,
            n_obs
    ):
        pred_cost_gain = tf.add(
            tf.einsum(
                'ni,in->n',
                neg_jac,
                proposed_vector
            ) / n_obs,
            0.5 * tf.einsum(
                'nix,xin->n',
                tf.einsum('inx,nij->njx',
                          tf.expand_dims(proposed_vector, axis=-1),
                          hessian_fim),
                tf.expand_dims(proposed_vector, axis=0)
            ) / tf.square(n_obs)
        )
        return pred_cost_gain

    def trust_region_linear_cost_gain(
            self,
            proposed_vector,
            neg_jac,
            n_obs
    ):
        pred_cost_gain = tf.reduce_sum(tf.multiply(
            proposed_vector,
            tf.transpose(neg_jac)
        ), axis=0)
        return pred_cost_gain

    def trust_region_ops(
            self,
            likelihood_container,
            proposed_vector,
            proposed_vector_container,
            proposed_gain,
            proposed_gain_container,
            radius_container,
            dtype
    ):
        # Load hyper-parameters:
        assert pkg_constants.TRUST_REGION_ETA0 < pkg_constants.TRUST_REGION_ETA1, \
            "eta0 must be smaller than eta1"
        assert pkg_constants.TRUST_REGION_ETA1 <= pkg_constants.TRUST_REGION_ETA2, \
            "eta1 must be smaller than or equal to eta2"
        assert pkg_constants.TRUST_REGION_T1 <= 1, "t1 must be smaller than 1"
        assert pkg_constants.TRUST_REGION_T2 >= 1, "t1 must be larger than 1"
        # Set trust region hyper-parameters
        eta0 = tf.constant(pkg_constants.TRUST_REGION_ETA0, dtype=dtype)
        eta1 = tf.constant(pkg_constants.TRUST_REGION_ETA1, dtype=dtype)
        eta2 = tf.constant(pkg_constants.TRUST_REGION_ETA2, dtype=dtype)
        t1 = tf.constant(pkg_constants.TRUST_REGION_T1, dtype=dtype)
        t2 = tf.constant(pkg_constants.TRUST_REGION_T2, dtype=dtype)
        upper_bound = tf.constant(pkg_constants.TRUST_REGION_UPPER_BOUND, dtype=dtype)

        # Phase I: Perform a trial update.
        # Propose parameter update:
        train_op_nr_tr_prev = tf.group(
            tf.compat.v1.assign(likelihood_container, self.full_data_model.norm_neg_log_likelihood_eval1)
        )
        train_op_x_step = tf.group(
            tf.compat.v1.assign(proposed_vector_container, proposed_vector),
            tf.compat.v1.assign(proposed_gain_container, proposed_gain)
        )
        train_op_trial_update = tf.group(
            tf.compat.v1.assign(self.model_vars.params, self.model_vars.params - proposed_vector)
        )

        # Phase II: Evaluate success of trial update and complete update cycle.
        # Include parameter updates only if update improves cost function:
        delta_f_actual = likelihood_container - self.full_data_model.norm_neg_log_likelihood_eval0
        delta_f_ratio = tf.divide(delta_f_actual, proposed_gain_container)

        # Compute parameter updates.
        update_theta = tf.logical_and(delta_f_actual > eta0, tf.logical_not(self.model_vars.converged))
        update_theta_numeric = tf.expand_dims(tf.cast(update_theta, dtype), axis=0)
        keep_theta_numeric = tf.ones_like(update_theta_numeric) - update_theta_numeric
        theta_new_nr_tr = tf.add(
            tf.multiply(self.model_vars.params + proposed_vector_container, keep_theta_numeric),  # old values
            tf.multiply(self.model_vars.params, update_theta_numeric)  # new values
        )

        train_op_update_params = tf.compat.v1.assign(self.model_vars.params, theta_new_nr_tr)
        train_op_update_status = tf.compat.v1.assign(self.model_vars.updated, update_theta)

        # Update trusted region accordingly:
        decrease_radius = tf.logical_or(
            delta_f_actual <= eta0,
            tf.logical_and(delta_f_ratio <= eta1, tf.logical_not(self.model_vars.converged))
        )
        increase_radius = tf.logical_and(
            delta_f_actual > eta0,
            tf.logical_and(delta_f_ratio > eta2, tf.logical_not(self.model_vars.converged))
        )
        keep_radius = tf.logical_and(tf.logical_not(decrease_radius),
                                     tf.logical_not(increase_radius))
        radius_update = tf.add_n([
            tf.multiply(t1, tf.cast(decrease_radius, dtype)),
            tf.multiply(t2, tf.cast(increase_radius, dtype)),
            tf.multiply(tf.ones_like(t1), tf.cast(keep_radius, dtype))
        ])
        radius_new = tf.minimum(tf.multiply(radius_container, radius_update), upper_bound)
        train_op_update_radius = tf.compat.v1.assign(radius_container, radius_new)

        train_ops = {
            "update": proposed_vector_container,
            "trial_op": tf.group(
                train_op_nr_tr_prev,
                train_op_x_step,
                train_op_trial_update
            ),
            "update_op": tf.group(
                train_op_update_params,
                train_op_update_status,
                train_op_update_radius
            )
        }

        return train_ops


class TrainerGraphGLM:
    """

    """
    model_vars: ModelVarsGLM
    model_vars_eval: ModelVarsGLM

    full_data_model: FullDataModelGraphGLM
    batched_data_model: BatchedDataModelGraphGLM

    gradient_graph: GradientGraphGLM
    gradients_batch: tf.Tensor
    gradients_full: tf.Tensor

    nr_update_full: tf.Tensor
    nr_update_batched: tf.Tensor
    nr_tr_update_full: tf.Tensor
    nr_tr_update_batched: tf.Tensor
    irls_update_full: tf.Tensor
    irls_update_batched: tf.Tensor
    irls_tr_update_full: tf.Tensor
    irls_tr_update_batched: tf.Tensor

    nr_tr_radius: Union[tf.Variable, None]
    nr_tr_pred_cost_gain_full: Union[tf.Tensor, None]
    nr_tr_pred_cost_gain_batched: Union[tf.Tensor, None]

    irls_tr_radius: Union[tf.Variable, None]
    irls_tr_pred_cost_gain_full: Union[tf.Tensor, None]
    irls_tr_pred_cost_gain_batched: Union[tf.Tensor, None]

    num_observations: int
    num_features: int
    num_design_loc_params: int
    num_design_scale_params: int
    num_loc_params: int
    num_scale_params: int
    batch_size: int

    session: tf.compat.v1.Session
    graph: tf.Graph

    def __init__(
            self,
            provide_optimizers,
            train_loc,
            train_scale,
            dtype
    ):
        with tf.name_scope("training_graphs"):
            global_step = tf.compat.v1.train.get_or_create_global_step()

            if (train_loc or train_scale) and self.batched_data_model is not None:
                logger.debug(" ** building batched trainers")
                trainer_batch = train_utils.MultiTrainer(
                    variables=self.model_vars.params,
                    gradients=self.gradients_batch,
                    newton_delta=self.nr_update_batched,
                    irls_delta=self.irls_update_batched,
                    irls_gd_delta=self.irls_gd_update_batched,
                    train_ops_nr_tr=self.train_ops_nr_tr_batched,
                    train_ops_irls_tr=self.train_ops_irls_tr_batched,
                    train_ops_irls_gd_tr=self.train_ops_irls_gd_tr_batched,
                    learning_rate=self.learning_rate,
                    global_step=global_step,
                    apply_gradients=lambda grad: tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad),
                    provide_optimizers=provide_optimizers,
                    name="batch_data_trainers"
                )
                batch_gradient = trainer_batch.plain_gradient_by_variable(self.model_vars.params)
                batch_gradient = tf.reduce_sum(tf.abs(batch_gradient), axis=0)
            else:
                trainer_batch = None
                batch_gradient = None

            if train_loc or train_scale:
                logger.debug(" ** building full trainers")
                trainer_full = train_utils.MultiTrainer(
                    variables=self.model_vars.params,
                    gradients=self.gradients_full,
                    newton_delta=self.nr_update_full,
                    irls_delta=self.irls_update_full,
                    irls_gd_delta=self.irls_gd_update_full,
                    train_ops_nr_tr=self.train_ops_nr_tr_full,
                    train_ops_irls_tr=self.train_ops_irls_tr_full,
                    train_ops_irls_gd_tr=self.train_ops_irls_gd_tr_full,
                    learning_rate=self.learning_rate,
                    global_step=global_step,
                    apply_gradients=lambda grad: tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad),
                    provide_optimizers=provide_optimizers,
                    name="full_data_trainers"
                )
                full_gradient = trainer_full.plain_gradient_by_variable(self.model_vars.params)
                full_gradient = tf.reduce_sum(tf.abs(full_gradient), axis=0)
            else:
                trainer_full = None
                full_gradient = None

        # # ### BFGS implementation using SciPy L-BFGS
        # with tf1.name_scope("bfgs"):
        #     feature_idx = tf1.placeholder(dtype="int64", shape=())
        #
        #     X_s = tf1.gather(X, feature_idx, axis=1)
        #     a_s = tf1.gather(a, feature_idx, axis=1)
        #     b_s = tf1.gather(b, feature_idx, axis=1)
        #
        #     model = BasicModelGraph(X_s, design_loc, design_scale, a_s, b_s, size_factors=size_factors)
        #
        #     trainer = tf1.contrib.opt.ScipyOptimizerInterface(
        #         model.loss,
        #         method='L-BFGS-B',
        #         options={'maxiter': maxiter})

        self.global_step = global_step

        self.trainer_batch = trainer_batch
        self.gradient = batch_gradient

        self.trainer_full = trainer_full
        self.full_gradient = full_gradient

        self.train_op = None

    @abc.abstractmethod
    def param_bounds(self):
        pass


class EstimatorGraphGLM(TFEstimatorGraph, NewtonGraphGLM, TrainerGraphGLM):
    """
    The estimator graphs are all graph necessary to perform parameter updates and to
    summarise a current parameter estimate.

    The estimator graph class is divided into the following major sub graphs:

        - The input pipeline: Feed data for parameter updates.
        -
    """
    X: Union[tf.Tensor, tf.SparseTensor]

    a_var: tf.Tensor
    b_var: tf.Tensor

    model_vars: ModelVarsGLM
    model_vars_eval: ModelVarsGLM

    noise_model: str

    def __init__(
            self,
            num_observations: int,
            num_features: int,
            num_design_loc_params: int,
            num_design_scale_params: int,
            num_loc_params: int,
            num_scale_params: int,
            graph: tf.Graph,
            batch_size: int,
            constraints_loc: np.ndarray,
            constraints_scale: np.ndarray,
            dtype: str
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
        :param graph: tf1.Graph
        :param constraints_loc: tensor (all parameters x dependent parameters) or None
            Tensor that encodes how complete parameter set which includes dependent
            parameters arises from indepedent parameters: all = <constraints, indep>.
            This tensor describes this relation for the mean model.
            This form of constraints is used in vector generalized linear models (VGLMs).
            Assumed to be an identity matrix if None.
        :param constraints_scale: tensor (all parameters x dependent parameters) or None
            Tensor that encodes how complete parameter set which includes dependent
            parameters arises from indepedent parameters: all = <constraints, indep>.
            This tensor describes this relation for the dispersion model.
            This form of constraints is used in vector generalized linear models (VGLMs).
            Assumed to be an identity matrix if None.
        """
        TFEstimatorGraph.__init__(
            self=self,
            graph=graph
        )

        self.num_observations = num_observations
        self.num_observations_tf = tf.cast(num_observations, dtype=dtype)
        self.num_features = num_features
        self.num_design_loc_params = num_design_loc_params
        self.num_design_scale_params = num_design_scale_params
        self.num_loc_params = num_loc_params
        self.num_scale_params = num_scale_params
        self.batch_size = batch_size
        self.batch_size_tf = tf.cast(batch_size, dtype=dtype)

        self.constraints_loc = self._set_constraints(
            constraints=constraints_loc,
            num_design_params=self.num_design_loc_params,
            dtype=dtype
        )
        self.constraints_scale = self._set_constraints(
            constraints=constraints_scale,
            num_design_params=self.num_design_scale_params,
            dtype=dtype
        )

        self.learning_rate = tf.compat.v1.placeholder(dtype, shape=(), name="learning_rate")

    def _run_trainer_init(
            self,
            provide_optimizers,
            train_loc,
            train_scale,
            dtype
    ):
        logger.debug(" * building gradient graph")
        self.gradient_graph = GradientGraphGLM(
            model_vars=self.model_vars,
            full_data_model=self.full_data_model,
            batched_data_model=self.batched_data_model,
            train_loc=train_loc,
            train_scale=train_scale
        )
        self.gradients_batch = self.gradient_graph.gradients_batch
        self.gradients_full = self.gradient_graph.gradients_full

        logger.debug(" * building newton-type update graph")
        NewtonGraphGLM.__init__(
            self=self,
            provide_optimizers=provide_optimizers,
            train_mu=train_loc,
            train_r=train_scale,
            dtype=dtype
        )

        logger.debug(" * building trainers")
        TrainerGraphGLM.__init__(
            self=self,
            provide_optimizers=provide_optimizers,
            train_loc=train_loc,
            train_scale=train_scale,
            dtype=dtype
        )

        with tf.name_scope("init_op"):
            self.init_op = tf.compat.v1.global_variables_initializer()
            self.init_ops = []

    def _set_out_var(
            self,
            feature_isnonzero,
            dtype
    ):
        # ### output values:
        #       override all-zero features with lower bound coefficients
        with tf.name_scope("output"):
            logger.debug(" ** Build training graph: output")
            bounds_min, bounds_max = self.param_bounds(dtype)

            param_nonzero_a_var = tf.broadcast_to(feature_isnonzero, [self.num_loc_params, self.num_features])
            alt_a = tf.broadcast_to(bounds_min["a_var"], [self.num_loc_params, self.num_features])
            a_var = tf.where(
                param_nonzero_a_var,
                self.model_vars.a_var,
                alt_a
            )

            param_nonzero_b_var = tf.broadcast_to(feature_isnonzero, [self.num_scale_params, self.num_features])
            alt_b = tf.broadcast_to(bounds_min["b_var"], [self.num_scale_params, self.num_features])
            b_var = tf.where(
                param_nonzero_b_var,
                self.model_vars.b_var,
                alt_b
            )

        self.a_var = a_var
        self.b_var = b_var

    def _set_constraints(
            self,
            constraints,
            num_design_params,
            dtype
    ):
        if constraints is None:
            return None
            #return tf1.eye(
            #    num_rows=tf1.constant(num_design_params, shape=(), dtype="int32"),
            #    dtype=dtype
            #)
        else:
            # Check if identity was supplied:
            if constraints.shape[0] == constraints.shape[1]:
                if np.sum(constraints - np.eye(constraints.shape[0], dtype=constraints.dtype)) < 1e-12:
                    return None

            assert constraints.shape[0] == num_design_params, "constraint dimension mismatch"
            return tf.cast(constraints, dtype=dtype)

    @abc.abstractmethod
    def param_bounds(self):
        pass
