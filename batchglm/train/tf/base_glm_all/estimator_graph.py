from typing import Union
import logging

import tensorflow as tf
import numpy as np
import xarray as xr

from .external import EstimatorGraphGLM, FullDataModelGraphGLM, BatchedDataModelGraphGLM, ModelVarsGLM
from .external import op_utils
from .external import pkg_constants

logger = logging.getLogger(__name__)


class FullDataModelGraph(FullDataModelGraphGLM):
    """
    Computational graph to evaluate negative binomial GLM metrics on full data set.

    Evaluate model and cost function, Jacobians, Hessians and Fisher information matrix.
    """

    def __init__(
            self,
            num_observations,
            sample_indices: tf.Tensor,
            fetch_fn,
            batch_size: Union[int, tf.Tensor],
            model_vars,
            constraints_loc,
            constraints_scale,
            noise_model,
            train_a,
            train_b,
            provide_fim,
            graph,
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
            raise ValueError("noise model not recognized")
        self.noise_model = noise_model

        logger.debug("building input pipeline")
        with tf.name_scope("input_pipeline"):
            dataset = tf.data.Dataset.from_tensor_slices(sample_indices)
            batched_data = dataset.batch(batch_size)
            batched_data = batched_data.map(fetch_fn, num_parallel_calls=pkg_constants.TF_NUM_THREADS)

            def map_sparse(idx, data):
                X_tensor_ls, design_loc_tensor, design_scale_tensor, size_factors_tensor = data
                if len(X_tensor_ls) > 1:
                    X_tensor = tf.SparseTensor(X_tensor_ls[0], X_tensor_ls[1], X_tensor_ls[2])
                    X_tensor = tf.cast(X_tensor, dtype=dtype)
                else:
                    X_tensor = X_tensor_ls[0]
                return idx, (X_tensor, design_loc_tensor, design_scale_tensor, size_factors_tensor)

            batched_data = batched_data.map(map_sparse, num_parallel_calls=pkg_constants.TF_NUM_THREADS)
            batched_data = batched_data.prefetch(1)

        logger.debug("building likelihood")
        with tf.name_scope("log_likelihood"):
            def init_fun():
                return tf.zeros([model_vars.n_features], dtype=dtype)

            def map_fun(idx, data) -> BasicModelGraph:
                X, design_loc, design_scale, size_factors = data
                basic_model = BasicModelGraph(
                    X=X,
                    design_loc=design_loc,
                    design_scale=design_scale,
                    constraints_loc=constraints_loc,
                    constraints_scale=constraints_scale,
                    a_var=model_vars.a_var,
                    b_var=model_vars.b_var,
                    dtype=dtype,
                    size_factors=size_factors)
                return basic_model

            def reduce_fun(old, new):
                return tf.add(old, new)

            log_likelihood = batched_data.reduce(
                initial_state=init_fun(),
                reduce_func=lambda old, new: reduce_fun(old, map_fun(new[0], new[1]).log_likelihood)
            )

            norm_log_likelihood = log_likelihood / tf.cast(tf.size(sample_indices), dtype=log_likelihood.dtype)
            norm_neg_log_likelihood = - norm_log_likelihood
            loss = tf.reduce_sum(norm_neg_log_likelihood)

        # Save attributes necessary for reinitialization:
        self.fetch_fn = fetch_fn
        self.batch_size = batch_size
        self.train_a = train_a
        self.train_b = train_b
        self.dtype = dtype
        self.constraints_loc = constraints_loc
        self.constraints_scale = constraints_scale
        self.num_observations = num_observations
        self.idx_train_loc = model_vars.idx_train_loc if train_a else np.array([])
        self.idx_train_scale = model_vars.idx_train_scale if train_b else np.array([])
        self.idx_train = np.sort(np.concatenate([self.idx_train_loc, self.idx_train_scale]))

        self.batched_data = batched_data
        self.sample_indices = sample_indices

        self.log_likelihood = log_likelihood
        self.norm_log_likelihood = norm_log_likelihood
        self.norm_neg_log_likelihood = norm_neg_log_likelihood
        self.loss = loss

        logger.debug("building jacobians")
        with tf.name_scope("jacobians"):
            # Jacobian of full model for reporting.
            jacobian_full = Jacobians(
                batched_data=self.batched_data,
                sample_indices=self.sample_indices,
                constraints_loc=self.constraints_loc,
                constraints_scale=self.constraints_scale,
                model_vars=model_vars,
                mode=pkg_constants.JACOBIAN_MODE,
                noise_model=noise_model,
                iterator=True,
                jac_a=True,
                jac_b=True,
                dtype=dtype
            )
            # Jacobian of submodel which is to be trained.
            if train_a and train_b:
                neg_jac_train = jacobian_full.neg_jac
            elif train_a and not train_b:
                neg_jac_train = jacobian_full.neg_jac_a
            elif not train_a and train_b:
                neg_jac_train = jacobian_full.neg_jac_b
            else:
                neg_jac_train = None

        self.jac = jacobian_full
        self.neg_jac_train = neg_jac_train

        logger.debug("building hessians")
        with tf.name_scope("hessians"):
            # Hessian of full model for reporting.
            hessians_full = Hessians(
                batched_data=self.batched_data,
                sample_indices=self.sample_indices,
                constraints_loc=self.constraints_loc,
                constraints_scale=self.constraints_scale,
                model_vars=model_vars,
                mode=pkg_constants.HESSIAN_MODE,
                noise_model=noise_model,
                iterator=True,
                hess_a=True,
                hess_b=True,
                dtype=dtype
            )
            # Hessian of submodel which is to be trained.
            if train_a and train_b:
                neg_hessians_train = hessians_full.neg_hessian
            elif train_a and not train_b:
                neg_hessians_train = hessians_full.neg_hessian_aa
            elif not train_a and train_b:
                neg_hessians_train = hessians_full.neg_hessian_bb
            else:
                neg_hessians_train = None

        logger.debug("building fim")
        with tf.name_scope("fim"):
            if provide_fim:
                fim_full = FIM(
                    batched_data=self.batched_data,
                    sample_indices=self.sample_indices,
                    constraints_loc=self.constraints_loc,
                    constraints_scale=self.constraints_scale,
                    model_vars=model_vars,
                    noise_model=noise_model,
                    iterator=True,
                    update_a=True,
                    update_b=True,
                    dtype=dtype
                )
            else:
                fim_full = None

        self.hessians = hessians_full
        self.neg_hessians_train = neg_hessians_train

        self.provide_fim = provide_fim
        self.fim = fim_full


class BatchedDataModelGraph(BatchedDataModelGraphGLM):
    """
    Basic computational graph to evaluate negative binomial GLM metrics on batched data set.

    Evaluate model and cost function and Jacobians, Hessians and Fisher information matrix.
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
            noise_model: str,
            train_a,
            train_b,
            provide_fim,
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
        :param dtype: Precision used in tensorflow.
        """
        if noise_model == "nb":
            from .external_nb import BasicModelGraph, Jacobians, Hessians, FIM
        else:
            raise ValueError("noise model not recognized")
        self.noise_model = noise_model

        with tf.name_scope("input_pipeline"):
            data_indices = tf.data.Dataset.from_tensor_slices((
                tf.range(num_observations, name="sample_index")
            ))
            training_data = data_indices.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=2 * batch_size))
            training_data = training_data.batch(batch_size, drop_remainder=True)
            training_data = training_data.map(tf.contrib.framework.sort)  # sort indices - TODO why?
            training_data = training_data.map(fetch_fn, num_parallel_calls=pkg_constants.TF_NUM_THREADS)
            training_data = training_data.prefetch(buffer_size)

            def map_sparse(idx, data_batch):
                X_tensor_ls, design_loc_tensor, design_scale_tensor, size_factors_tensor = data_batch
                if len(X_tensor_ls) > 1:
                    X_tensor = tf.SparseTensor(X_tensor_ls[0], X_tensor_ls[1], X_tensor_ls[2])
                    X_tensor = tf.cast(X_tensor, dtype=dtype)
                else:
                    X_tensor = X_tensor_ls[0]
                return idx, (X_tensor, design_loc_tensor, design_scale_tensor, size_factors_tensor)

            training_data = training_data.map(map_sparse, num_parallel_calls=pkg_constants.TF_NUM_THREADS)
            iterator = training_data.make_one_shot_iterator()  # tf.compat.v1.data.make_one_shot_iterator(training_data) TODO: replace with tf>=v1.13

            batch_sample_index, batch_data = iterator.get_next()
            (batch_X, batch_design_loc, batch_design_scale, batch_size_factors) = batch_data

            batched_model = BasicModelGraph(
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

        # Save hyper parameters to be reused for reinitialization:
        self.num_observations = num_observations
        self.fetch_fn = fetch_fn
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.train_a = train_a
        self.train_b = train_b
        self.dtype = dtype

        self.X = batched_model.X
        self.design_loc = batched_model.design_loc
        self.design_scale = batched_model.design_scale
        self.constraints_loc = batched_model.constraints_loc
        self.constraints_scale = batched_model.constraints_scale
        self.idx_train_loc = model_vars.idx_train_loc if train_a else np.array([])
        self.idx_train_scale = model_vars.idx_train_scale if train_b else np.array([])
        self.idx_train = np.sort(np.concatenate([self.idx_train_loc, self.idx_train_scale]))

        self.batched_model = batched_model
        self.batched_data = batch_data
        self.sample_indices = batch_sample_index

        self.mu = batched_model.mu
        self.r = batched_model.r
        self.sigma2 = batched_model.sigma2

        self.probs = batched_model.probs
        self.log_probs = batched_model.log_probs

        self.log_likelihood = batched_model.log_likelihood
        self.norm_log_likelihood = batched_model.norm_log_likelihood
        self.norm_neg_log_likelihood = batched_model.norm_neg_log_likelihood
        self.loss = batched_model.loss

        # Define the jacobian on the batched model for newton-rhapson:
        # (note that these are the Jacobian matrix blocks
        # of the trained subset of parameters).
        if train_a or train_b:
            batch_jac = Jacobians(
                batched_data=self.batched_data,
                sample_indices=self.sample_indices,
                constraints_loc=self.constraints_loc,
                constraints_scale=self.constraints_scale,
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

        self.jac_train = batch_jac

        # Define the hessian on the batched model for newton-rhapson:
        # (note that these are the Hessian matrix blocks
        # of the trained subset of parameters).
        if train_a or train_b:
            batch_hessians = Hessians(
                batched_data=self.batched_data,
                sample_indices=self.sample_indices,
                constraints_loc=self.constraints_loc,
                constraints_scale=self.constraints_scale,
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
        if (train_a or train_b) and provide_fim:
            batch_fim = FIM(
                batched_data=self.batched_data,
                sample_indices=self.sample_indices,
                constraints_loc=self.constraints_loc,
                constraints_scale=self.constraints_scale,
                model_vars=model_vars,
                noise_model=noise_model,
                iterator=False,
                update_a=train_a,
                update_b=train_b,
                dtype=dtype
            )
        else:
            batch_fim = None

        self.hessians_train = batch_hessians
        self.provide_fim = provide_fim
        self.fim_train = batch_fim


class EstimatorGraphAll(EstimatorGraphGLM):
    """

    Contains model_vars, full_data_model and batched_data_model which are the
    primary training objects. All three also exist as *_eval which can be used
    to perform and iterative optmization within a single parameter update, such
    as during a line search.
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
            graph: tf.Graph,
            batch_size: int,
            init_a,
            init_b,
            train_loc: bool,
            train_scale: bool,
            provide_optimizers: Union[dict, None],
            provide_batched: bool,
            termination_type: str,
            extended_summary: bool,
            noise_model: str,
            dtype: str
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

            logger.debug("building models variables")
            with tf.name_scope("model_vars"):
                self.model_vars = ModelVars(
                    dtype=dtype,
                    init_a=init_a,
                    init_b=init_b,
                    constraints_loc=self.constraints_loc,
                    constraints_scale=self.constraints_scale
                )
                self.idx_nonconverged = np.where(np.logical_not(self.model_vars.converged))[0]

            # ### performance related settings
            buffer_size = 4

            # Check whether it is necessary to compute FIM:
            # The according sub-graphs are only compiled if this is needed during training.
            if provide_optimizers["irls"] or provide_optimizers["irls_tr"]:
                provide_fim = True
            else:
                provide_fim = False

            with tf.name_scope("batched_data"):
                logger.debug("building batched data model")
                if provide_batched:
                    self.batched_data_model = BatchedDataModelGraph(
                        num_observations=self.num_observations,
                        fetch_fn=fetch_fn,
                        batch_size=batch_size,
                        buffer_size=buffer_size,
                        model_vars=self.model_vars,
                        constraints_loc=self.constraints_loc,
                        constraints_scale=self.constraints_scale,
                        train_a=train_loc,
                        train_b=train_scale,
                        noise_model=noise_model,
                        provide_fim=provide_fim,
                        dtype=dtype
                    )
                else:
                    self.batched_data_model = None

            with tf.name_scope("full_data"):
                logger.debug("building full data model")
                # ### alternative definitions for custom observations:
                sample_selection = tf.placeholder_with_default(
                    tf.range(num_observations),
                    shape=(None,),
                    name="sample_selection"
                )
                self.full_data_model = FullDataModelGraph(
                    num_observations=self.num_observations,
                    sample_indices=sample_selection,
                    fetch_fn=fetch_fn,
                    batch_size=batch_size,
                    model_vars=self.model_vars,
                    constraints_loc=self.constraints_loc,
                    constraints_scale=self.constraints_scale,
                    train_a=train_loc,
                    train_b=train_scale,
                    noise_model=noise_model,
                    provide_fim=provide_fim,
                    graph=self.graph,
                    dtype=dtype
                )

            logger.debug("building trainers")
            self._run_trainer_init(
                termination_type=termination_type,
                provide_optimizers=provide_optimizers,
                train_loc=train_loc,
                train_scale=train_scale,
                dtype=dtype
            )

            # Define output metrics:
            logger.debug("building outputs")
            self.test = tf.constant(0)
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
            if extended_summary:
                tf.summary.histogram('a_var', self.model_vars.a_var)
                tf.summary.histogram('b_var', self.model_vars.b_var)
                tf.summary.scalar('loss', self.full_data_model.loss)
                tf.summary.scalar('learning_rate', self.learning_rate)

        self.saver = tf.train.Saver()
        self.merged_summary = tf.summary.merge_all()
