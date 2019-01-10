from typing import Union
import logging

import tensorflow as tf
import numpy as np
import xarray as xr

from .external import GradientGraphGLM, NewtonGraphGLM, TrainerGraphGLM
from .external import EstimatorGraphGLM, FullDataModelGraphGLM, BatchedDataModelGraphGLM
from .external import op_utils
from .external import pkg_constants

logger = logging.getLogger(__name__)


class FullDataModelGraph(FullDataModelGraphGLM):
    """
    Computational graph to evaluate negative binomial GLM metrics on full data set.
    """

    def __init__(
            self,
            sample_indices: tf.Tensor,
            fetch_fn,
            batch_size: Union[int, tf.Tensor],
            model_vars,
            constraints_loc,
            constraints_scale,
            train_a,
            train_b,
            noise_model: str,
            dtype
    ):
        """
        :param sample_indices:
            TODO
        :param fetch_fn:
            TODO
        :param batch_size: int
            Size of mini-batches used.
        :param model_vars: ModelVars
            Variables of model. Contains tf.Variables which are optimized.
        :param constraints_loc: tensor (all parameters x dependent parameters)
            Tensor that encodes how complete parameter set which includes dependent
            parameters arises from indepedent parameters: all = <constraints, indep>.
            This tensor describes this relation for the mean model.
            This form of constraints is used in vector generalized linear models (VGLMs).
        :param constraints_scale: tensor (all parameters x dependent parameters)
            Tensor that encodes how complete parameter set which includes dependent
            parameters arises from indepedent parameters: all = <constraints, indep>.
            This tensor describes this relation for the dispersion model.
            This form of constraints is used in vector generalized linear models (VGLMs).
        :param train_mu: bool
            Whether to train mean model. If False, the initialisation is kept.
        :param train_r: bool
            Whether to train dispersion model. If False, the initialisation is kept.
        :param dtype: Precision used in tensorflow.
        """
        if noise_model == "nb":
            from .external_nb import BasicModelGraph, Jacobians, Hessians, FIM
        else:
            raise ValueError("noise model not rewcognized")
        self.noise_model = noise_model

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
                a_var=model_vars.a_var,
                b_var=model_vars.b_var,
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
            # Hessian of full model for reporting.
            hessians_full = Hessians(
                batched_data=batched_data,
                sample_indices=sample_indices,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                model_vars=model_vars,
                mode=pkg_constants.HESSIAN_MODE,
                noise_model=noise_model,
                iterator=True,
                hess_a=True,
                hess_b=True,
                dtype=dtype
            )
            # Hessian of submodel which is to be trained.
            if train_a or train_b:
                if not train_a or not train_b:
                    hessians_train = Hessians(
                        batched_data=batched_data,
                        sample_indices=sample_indices,
                        constraints_loc=constraints_loc,
                        constraints_scale=constraints_scale,
                        model_vars=model_vars,
                        mode=pkg_constants.HESSIAN_MODE,
                        noise_model=noise_model,
                        iterator=True,
                        hess_a=train_a,
                        hess_b=train_b,
                        dtype=dtype
                    )
                else:
                    hessians_train = hessians_full
            else:
                hessians_train = None

            fim_full = FIM(
                batched_data=batched_data,
                sample_indices=sample_indices,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                model_vars=model_vars,
                mode=pkg_constants.HESSIAN_MODE,
                noise_model=noise_model,
                iterator=True,
                update_a=True,
                update_b=True,
                dtype=dtype
            )
            # Fisher information matrix of submodel which is to be trained.
            if train_a or train_b:
                if not train_a or not train_b:
                    fim_train = FIM(
                        batched_data=batched_data,
                        sample_indices=sample_indices,
                        constraints_loc=constraints_loc,
                        constraints_scale=constraints_scale,
                        model_vars=model_vars,
                        mode=pkg_constants.HESSIAN_MODE,
                        noise_model=noise_model,
                        iterator=True,
                        update_a=train_a,
                        update_b=train_b,
                        dtype=dtype
                    )
                else:
                    fim_train = fim_full
            else:
                fim_train = None

        with tf.name_scope("jacobians"):
            # Jacobian of full model for reporting.
            jacobian_full = Jacobians(
                batched_data=batched_data,
                sample_indices=sample_indices,
                batch_model=None,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                model_vars=model_vars,
                mode=pkg_constants.JACOBIAN_MODE,
                noise_model=noise_model,
                iterator=True,
                jac_a=True,
                jac_b=True,
                dtype=dtype
            )
            # Jacobian of submodel which is to be trained.
            if train_a or train_b:
                if not train_a or not train_b:
                    jacobian_train = Jacobians(
                        batched_data=batched_data,
                        sample_indices=sample_indices,
                        batch_model=None,
                        constraints_loc=constraints_loc,
                        constraints_scale=constraints_scale,
                        model_vars=model_vars,
                        mode=pkg_constants.JACOBIAN_MODE,
                        noise_model=noise_model,
                        iterator=True,
                        jac_a=train_a,
                        jac_b=train_b,
                        dtype=dtype
                    )
                else:
                    jacobian_train = jacobian_full
            else:
                jacobian_train = None

        self.X = model.X
        self.design_loc = model.design_loc
        self.design_scale = model.design_scale

        self.batched_data = batched_data

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

        self.jac = jacobian_full.jac
        self.jac_train = jacobian_train

        self.hessians = hessians_full
        self.hessians_train = hessians_train

        self.fim = fim_full
        self.fim_train = fim_train


class BatchedDataModelGraph(BatchedDataModelGraphGLM):
    """
    Computational graph to evaluate negative binomial GLM metrics on batched data set.
    """

    def __init__(
            self,
            num_observations,
            fetch_fn,
            batch_size: Union[int, tf.Tensor],
            buffer_size: int,
            model_vars,
            constraints_loc,
            constraints_scale,
            train_a,
            train_b,
            noise_model: str,
            dtype
    ):
        """
        :param fetch_fn:
            TODO
        :param batch_size: int
            Size of mini-batches used.
        :param model_vars: ModelVars
            Variables of model. Contains tf.Variables which are optimized.
        :param constraints_loc: tensor (all parameters x dependent parameters)
            Tensor that encodes how complete parameter set which includes dependent
            parameters arises from indepedent parameters: all = <constraints, indep>.
            This tensor describes this relation for the mean model.
            This form of constraints is used in vector generalized linear models (VGLMs).
        :param constraints_scale: tensor (all parameters x dependent parameters)
            Tensor that encodes how complete parameter set which includes dependent
            parameters arises from indepedent parameters: all = <constraints, indep>.
            This tensor describes this relation for the dispersion model.
            This form of constraints is used in vector generalized linear models (VGLMs).
        :param train_mu: bool
            Whether to train mean model. If False, the initialisation is kept.
        :param train_r: bool
            Whether to train dispersion model. If False, the initialisation is kept.
        :param dtype: Precision used in tensorflow.
        """
        if noise_model == "nb":
            from .external_nb import BasicModelGraph, Jacobians, Hessians, FIM
        else:
            raise ValueError("noise model not rewcognized")
        self.noise_model = noise_model


        with tf.name_scope("input_pipeline"):
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

        with tf.name_scope("batch"):
            batch_model = BasicModelGraph(
                X=batch_X,
                design_loc=batch_design_loc,
                design_scale=batch_design_scale,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                a_var=model_vars.a_var,
                b_var=model_vars.b_var,
                dtype=dtype,
                size_factors=batch_size_factors
            )

            # Define the jacobian on the batched model for newton-rhapson:
            # (note that these are the Jacobian matrix blocks
            # of the trained subset of parameters).
            if train_a or train_b:
                batch_jac = Jacobians(
                    batched_data=batch_data,
                    sample_indices=batch_sample_index,
                    batch_model=batch_model,
                    constraints_loc=constraints_loc,
                    constraints_scale=constraints_scale,
                    model_vars=model_vars,
                    mode=pkg_constants.JACOBIAN_MODE,
                    noise_model=noise_model,
                    iterator=False,
                    jac_a=train_a,
                    jac_b=train_b,
                    dtype=dtype
                )
            else:
                batch_jac = None

            # Define the hessian on the batched model for newton-rhapson:
            # (note that these are the Hessian matrix blocks
            # of the trained subset of parameters).
            if train_a or train_b:
                batch_hessians = Hessians(
                    batched_data=batch_data,
                    sample_indices=batch_sample_index,
                    constraints_loc=constraints_loc,
                    constraints_scale=constraints_scale,
                    model_vars=model_vars,
                    mode=pkg_constants.HESSIAN_MODE,
                    noise_model=noise_model,
                    iterator=False,
                    hess_a=train_a,
                    hess_b=train_b,
                    dtype=dtype
                )
            else:
                batch_hessians = None

            # Define the IRLS components on the batched model:
            # (note that these are the IRLS matrix blocks
            # of the trained subset of parameters).
            if train_a or train_b:
                batch_fim = FIM(
                    batched_data=batch_data,
                    sample_indices=batch_sample_index,
                    constraints_loc=constraints_loc,
                    constraints_scale=constraints_scale,
                    model_vars=model_vars,
                    mode=pkg_constants.HESSIAN_MODE,
                    noise_model=noise_model,
                    iterator=False,
                    update_a=train_a,
                    update_b=train_b,
                    dtype=dtype
                )
            else:
                batch_fim = None

        self.X = batch_model.X
        self.design_loc = batch_model.design_loc
        self.design_scale = batch_model.design_scale

        self.batched_data = batch_data

        self.mu = batch_model.mu
        self.r = batch_model.r
        self.sigma2 = batch_model.sigma2

        self.probs = batch_model.probs
        self.log_probs = batch_model.log_probs

        self.sample_indices = batch_sample_index

        self.log_likelihood = batch_model.log_likelihood
        self.norm_log_likelihood = batch_model.norm_log_likelihood
        self.norm_neg_log_likelihood = batch_model.norm_neg_log_likelihood
        self.loss = batch_model.loss

        self.jac_train = batch_jac
        self.hessians_train = batch_hessians
        self.fim_train = batch_fim


class EstimatorGraphAll(EstimatorGraphGLM):
    """

    """
    mu: tf.Tensor
    sigma2: tf.Tensor

    def __init__(
            self,
            fetch_fn,
            feature_isnonzero,
            num_observations,
            num_features,
            num_design_loc_params,
            num_design_scale_params,
            num_loc_params,
            num_scale_params,
            constraints_loc: xr.DataArray,
            constraints_scale: xr.DataArray,
            graph: tf.Graph = None,
            batch_size: int = None,
            init_a=None,
            init_b=None,
            train_loc: bool = True,
            train_scale: bool = True,
            provide_optimizers: Union[dict, None] = None,
            termination_type: str = "global",
            extended_summary=False,
            noise_model: str = None,
            dtype="float32"
    ):
        """

        :param fetch_fn:
            TODO
        :param feature_isnonzero:
            Whether all observations of a feature are zero. Features for which this
            is the case are not fitted.
        :param num_observations: int
            Number of observations.
        :param num_features: int
            Number of features.
        :param num_design_loc_params: int
            Number of parameters per feature in mean model.
        :param num_design_scale_params: int
            Number of parameters per feature in scale model.
        :param graph: tf.Graph
        :param batch_size: int
            Size of mini-batches used.
        :param init_a: nd.array (mean model size x features)
            Initialisation for all parameters of mean model.
        :param init_b: nd.array (dispersion model size x features)
            Initialisation for all parameters of dispersion model.
        :param constraints_loc: tensor (all parameters x dependent parameters)
            Tensor that encodes how complete parameter set which includes dependent
            parameters arises from indepedent parameters: all = <constraints, indep>.
            This tensor describes this relation for the mean model.
            This form of constraints is used in vector generalized linear models (VGLMs).
        :param constraints_scale: tensor (all parameters x dependent parameters)
            Tensor that encodes how complete parameter set which includes dependent
            parameters arises from indepedent parameters: all = <constraints, indep>.
            This tensor describes this relation for the dispersion model.
            This form of constraints is used in vector generalized linear models (VGLMs).
        :param train_loc: bool
            Whether to train mean model. If False, the initialisation is kept.
        :param train_scale: bool
            Whether to train dispersion model. If False, the initialisation is kept.
        :param provide_optimizers:
        :param termination_type:
        :param extended_summary:
        :param dtype: Precision used in tensorflow.
        """
        if noise_model == "nb":
            from .external_nb import BasicModelGraph, ModelVars, Jacobians, Hessians, FIM
        else:
            raise ValueError("noise model not recognized")
        self.noise_model = noise_model

        EstimatorGraphGLM.__init__(
            self=self,
            num_observations=num_observations,
            num_features=num_features,
            num_design_loc_params=num_design_loc_params,
            num_design_scale_params=num_design_scale_params,
            num_loc_params=num_loc_params,
            num_scale_params=num_scale_params,
            graph=graph,
            batch_size=batch_size,
            constraints_loc=constraints_loc,
            constraints_scale=constraints_scale,
            dtype=dtype
        )

        # initial graph elements
        with self.graph.as_default():

            with tf.name_scope("model_vars"):
                self.model_vars = ModelVars(
                    dtype=dtype,
                    init_a=init_a,
                    init_b=init_b,
                    constraints_loc=constraints_loc,
                    constraints_scale=constraints_scale
                )
                self.idx_nonconverged = np.where(self.model_vars.converged == False)[0]

            # ### performance related settings
            buffer_size = 4

            with tf.name_scope("batched_data"):
                logger.debug(" ** Build batched data model")
                self.batched_data_model = BatchedDataModelGraph(
                    num_observations=self.num_observations,
                    fetch_fn=fetch_fn,
                    batch_size=batch_size,
                    buffer_size=buffer_size,
                    model_vars=self.model_vars,
                    constraints_loc=constraints_loc,
                    constraints_scale=constraints_scale,
                    train_a=train_loc,
                    train_b=train_scale,
                    noise_model=noise_model,
                    dtype=dtype
                )

            with tf.name_scope("full_data"):
                logger.debug(" ** Build full data model")
                # ### alternative definitions for custom observations:
                sample_selection = tf.placeholder_with_default(
                    tf.range(num_observations),
                    shape=(None,),
                    name="sample_selection"
                )
                self.full_data_model = FullDataModelGraph(
                    sample_indices=sample_selection,
                    fetch_fn=fetch_fn,
                    batch_size=batch_size * buffer_size,
                    model_vars=self.model_vars,
                    constraints_loc=constraints_loc,
                    constraints_scale=constraints_scale,
                    train_a=train_loc,
                    train_b=train_scale,
                    noise_model=noise_model,
                    dtype=dtype
                )

            self._run_trainer_init(
                termination_type=termination_type,
                provide_optimizers=provide_optimizers,
                train_loc=train_loc,
                train_scale=train_scale,
                dtype=dtype
            )

            # Define output metrics:
            self._set_out_var(
                feature_isnonzero=feature_isnonzero,
                dtype=dtype
            )
            self.loss = self.full_data_model.loss
            self.log_likelihood = self.full_data_model.log_likelihood
            self.hessians = self.full_data_model.hessians.hessian
            self.fisher_inv = op_utils.pinv(self.full_data_model.hessians.neg_hessian)  # TODO switch for fim?
            # Summary statistics on feature-wise model gradients:
            self.gradients = tf.reduce_sum(tf.transpose(self.gradients_full), axis=1)

        with tf.name_scope('summaries'):
            tf.summary.histogram('a_var', self.model_vars.a_var)
            tf.summary.histogram('b_var', self.model_vars.b_var)
            tf.summary.scalar('loss', self.batched_data_model.loss)
            tf.summary.scalar('learning_rate', self.learning_rate)

            if extended_summary:
                pass

        self.saver = tf.train.Saver()
        self.merged_summary = tf.summary.merge_all()
