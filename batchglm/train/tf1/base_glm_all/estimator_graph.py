import logging
import numpy as np
import tensorflow as tf
from typing import Union

from .external import EstimatorGraphGLM, FullDataModelGraphGLM, BatchedDataModelGraphGLM, ModelVarsGLM
from .external import pkg_constants

logger = logging.getLogger(__name__)


class FullDataModelGraph(FullDataModelGraphGLM):
    """
    Computational graph to evaluate GLM metrics on full data set.

    Evaluate model and cost function, Jacobians, Hessians and Fisher information matrix.
    """

    def __init__(
            self,
            num_observations,
            sample_indices: tf.Tensor,
            fetch_fn,
            batch_size: Union[int, tf.Tensor],
            model_vars: ModelVarsGLM,
            constraints_loc,
            constraints_scale,
            noise_model,
            train_a,
            train_b,
            compute_fim,
            compute_hessian,
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
            Variables of model. Contains tf1.Variables which are optimized.
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
            from .external_nb import ReducibleTensors
        elif noise_model == "norm":
            from .external_norm import ReducibleTensors
        elif noise_model == "beta":
            from .external_beta import ReducibleTensors
        else:
            raise ValueError("noise model not recognized")
        self.noise_model = noise_model

        logger.debug("building input pipeline")
        with tf.name_scope("input_pipeline"):
            data_set = tf.data.Dataset.from_tensor_slices(sample_indices)
            data_set = data_set.batch(batch_size)
            data_set = data_set.map(fetch_fn, num_parallel_calls=pkg_constants.TF_NUM_THREADS)

            def map_sparse(idx, data):
                X_tensor_ls, design_loc_tensor, design_scale_tensor, size_factors_tensor = data
                if len(X_tensor_ls) > 1:
                    X_tensor = tf.SparseTensor(X_tensor_ls[0], X_tensor_ls[1], X_tensor_ls[2])
                    X_tensor = tf.cast(X_tensor, dtype=dtype)
                else:
                    X_tensor = X_tensor_ls[0]
                return idx, (X_tensor, design_loc_tensor, design_scale_tensor, size_factors_tensor)

            data_set = data_set.map(map_sparse, num_parallel_calls=pkg_constants.TF_NUM_THREADS)
            data_set = data_set.prefetch(1)

        with tf.name_scope("reducible_tensors_train"):
            reducibles_train = ReducibleTensors(
                model_vars=model_vars,
                noise_model=noise_model,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                sample_indices=sample_indices,
                data_set=data_set,
                data_batch=None,
                mode_jac=pkg_constants.JACOBIAN_MODE,
                mode_hessian=pkg_constants.HESSIAN_MODE,
                mode_fim=pkg_constants.FIM_MODE,
                compute_a=train_a,
                compute_b=train_b,
                compute_jac=True,
                compute_hessian=compute_hessian,
                compute_fim=compute_fim,
                compute_ll=False
            )
            self.neg_jac_train = reducibles_train.neg_jac_train
            self.jac = reducibles_train.jac
            self.neg_jac_a = reducibles_train.neg_jac_a
            self.neg_jac_b = reducibles_train.neg_jac_b
            self.jac_b = reducibles_train.jac_b

            self.hessians = reducibles_train.hessian
            self.neg_hessians_train = reducibles_train.neg_hessian_train

            self.fim_a = reducibles_train.fim_a
            self.fim_b = reducibles_train.fim_b

            self.train_set = reducibles_train.set

        with tf.name_scope("reducible_tensors_finalize"):
            reducibles_finalize = ReducibleTensors(
                model_vars=model_vars,
                noise_model=noise_model,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                sample_indices=sample_indices,
                data_set=data_set,
                data_batch=None,
                mode_jac=pkg_constants.JACOBIAN_MODE,
                mode_hessian=pkg_constants.HESSIAN_MODE,
                mode_fim=pkg_constants.FIM_MODE,
                compute_a=True,
                compute_b=True,
                compute_jac=True,
                compute_hessian=True,
                compute_fim=False,
                compute_ll=True
            )
            self.hessians_final = reducibles_finalize.hessian
            self.neg_jac_final = reducibles_finalize.neg_jac
            self.log_likelihood_final = reducibles_finalize.ll
            self.loss_final = tf.reduce_sum(-self.log_likelihood_final / num_observations)

            self.final_set = reducibles_finalize.set

        with tf.name_scope("reducible_tensors_eval_ll"):
            reducibles_eval0 = ReducibleTensors(
                model_vars=model_vars,
                noise_model=noise_model,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                sample_indices=sample_indices,
                data_set=data_set,
                data_batch=None,
                mode_jac=pkg_constants.JACOBIAN_MODE,
                mode_hessian=pkg_constants.HESSIAN_MODE,
                mode_fim=pkg_constants.FIM_MODE,
                compute_a=train_a,
                compute_b=train_b,
                compute_jac=False,
                compute_hessian=False,
                compute_fim=False,
                compute_ll=True
            )
            self.log_likelihood_eval0 = reducibles_eval0.ll
            self.norm_neg_log_likelihood_eval0 = -self.log_likelihood_eval0 / num_observations
            self.loss_eval0 = tf.reduce_sum(self.norm_neg_log_likelihood_eval0)

            self.eval0_set = reducibles_eval0.set

        with tf.name_scope("reducible_tensors_eval_ll_jac"):
            reducibles_eval1 = ReducibleTensors(
                model_vars=model_vars,
                noise_model=noise_model,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                sample_indices=sample_indices,
                data_set=data_set,
                data_batch=None,
                mode_jac=pkg_constants.JACOBIAN_MODE,
                mode_hessian=pkg_constants.HESSIAN_MODE,
                mode_fim=pkg_constants.FIM_MODE,
                compute_a=train_a,
                compute_b=train_b,
                compute_jac=True,
                compute_hessian=False,
                compute_fim=False,
                compute_ll=True
            )
            self.log_likelihood_eval1 = reducibles_eval1.ll
            self.norm_neg_log_likelihood_eval1 = -self.log_likelihood_eval1 / num_observations
            self.loss_eval1 = tf.reduce_sum(self.norm_neg_log_likelihood_eval1)
            self.neg_jac_train_eval = reducibles_eval1.neg_jac_train

            self.eval1_set = reducibles_eval1.set

        self.num_observations = num_observations
        self.idx_train_loc = model_vars.idx_train_loc if train_a else np.array([])
        self.idx_train_scale = model_vars.idx_train_scale if train_b else np.array([])
        self.idx_train = np.sort(np.concatenate([self.idx_train_loc, self.idx_train_scale]))


class BatchedDataModelGraph(BatchedDataModelGraphGLM):
    """
    Basic computational graph to evaluate GLM metrics on batched data set.

    Evaluate model and cost function and Jacobians, Hessians and Fisher information matrix.
    """

    def __init__(
            self,
            num_observations,
            fetch_fn,
            batch_size: Union[int, tf.Tensor],
            buffer_size: int,
            model_vars: ModelVarsGLM,
            constraints_loc,
            constraints_scale,
            noise_model: str,
            train_a,
            train_b,
            compute_fim,
            compute_hessian,
            dtype
    ):
        """
        :param fetch_fn:
            TODO
        :param batch_size: int
            Size of mini-batches used.
        :param model_vars: ModelVars
            Variables of model. Contains tf1.Variables which are optimized.
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
            from .external_nb import ReducibleTensors
        elif noise_model == "norm":
            from .external_norm import ReducibleTensors
        elif noise_model == "beta":
            from .external_beta import ReducibleTensors
        else:
            raise ValueError("noise model not recognized")
        self.noise_model = noise_model

        with tf.name_scope("input_pipeline"):
            data_set = tf.data.Dataset.from_tensor_slices((
                tf.range(num_observations, name="sample_index")
            ))
            data_set = data_set.shuffle(buffer_size=2 * batch_size)
            data_set = data_set.repeat()
            data_set = data_set.batch(batch_size, drop_remainder=True)
            data_set = data_set.map(tf.contrib.framework.sort)  # sort indices - TODO why?
            data_set = data_set.map(fetch_fn, num_parallel_calls=pkg_constants.TF_NUM_THREADS)
            data_set = data_set.prefetch(buffer_size)

            def map_sparse(idx, data_batch):
                X_tensor_ls, design_loc_tensor, design_scale_tensor, size_factors_tensor = data_batch
                if len(X_tensor_ls) > 1:
                    X_tensor = tf.SparseTensor(X_tensor_ls[0], X_tensor_ls[1], X_tensor_ls[2])
                    X_tensor = tf.cast(X_tensor, dtype=dtype)
                else:
                    X_tensor = X_tensor_ls[0]
                return idx, (X_tensor, design_loc_tensor, design_scale_tensor, size_factors_tensor)

            data_set = data_set.map(map_sparse, num_parallel_calls=pkg_constants.TF_NUM_THREADS)
            iterator = tf.compat.v1.data.make_one_shot_iterator(data_set)

            batch_sample_index, batch_data = iterator.get_next()

        with tf.name_scope("reducible_tensors_train"):
            reducibles_train = ReducibleTensors(
                model_vars=model_vars,
                noise_model=noise_model,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                sample_indices=batch_sample_index,
                data_set=None,
                data_batch=batch_data,
                mode_jac=pkg_constants.JACOBIAN_MODE,
                mode_hessian=pkg_constants.HESSIAN_MODE,
                mode_fim=pkg_constants.FIM_MODE,
                compute_a=train_a,
                compute_b=train_b,
                compute_jac=True,
                compute_hessian=compute_hessian,
                compute_fim=compute_fim,
                compute_ll=False
            )

            self.neg_jac_train = reducibles_train.neg_jac_train
            self.jac = reducibles_train.jac
            self.neg_jac_a = reducibles_train.neg_jac_a
            self.neg_jac_b = reducibles_train.neg_jac_b
            self.jac_b = reducibles_train.jac_b

            self.hessians = reducibles_train.hessian
            self.neg_hessians_train = reducibles_train.neg_hessian_train

            self.fim_a = reducibles_train.fim_a
            self.fim_b = reducibles_train.fim_b

            self.train_set = reducibles_train.set

        with tf.name_scope("reducible_tensors_eval"):
            reducibles_eval = ReducibleTensors(
                model_vars=model_vars,
                noise_model=noise_model,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                sample_indices=batch_sample_index,
                data_set=None,
                data_batch=batch_data,
                mode_jac=pkg_constants.JACOBIAN_MODE,
                mode_hessian=pkg_constants.HESSIAN_MODE,
                mode_fim=pkg_constants.FIM_MODE,
                compute_a=True,
                compute_b=True,
                compute_jac=True,
                compute_hessian=False,
                compute_fim=False,
                compute_ll=True
            )

            self.log_likelihood = reducibles_eval.ll
            self.norm_log_likelihood = self.log_likelihood / num_observations
            self.norm_neg_log_likelihood = -self.norm_log_likelihood
            self.loss = tf.reduce_sum(self.norm_neg_log_likelihood)

            self.neg_jac_train_eval = reducibles_train.neg_jac_train

            self.eval_set = reducibles_eval.set

        self.num_observations = num_observations
        self.idx_train_loc = model_vars.idx_train_loc if train_a else np.array([])
        self.idx_train_scale = model_vars.idx_train_scale if train_b else np.array([])
        self.idx_train = np.sort(np.concatenate([self.idx_train_loc, self.idx_train_scale]))


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
            constraints_loc: np.ndarray,
            constraints_scale: np.ndarray,
            graph: tf.Graph,
            batch_size: int,
            init_a,
            init_b,
            train_loc: bool,
            train_scale: bool,
            provide_optimizers: Union[dict, None],
            provide_batched: bool,
            provide_hessian: bool,
            provide_fim: bool,
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
        :param graph: tf1.Graph
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
        :param extended_summary:
        :param dtype: Precision used in tensorflow.
        """
        if noise_model == "nb":
            from .external_nb import ModelVars
        elif noise_model == "norm":
            from .external_norm import ModelVars
        elif noise_model == "beta":
            from .external_beta import ModelVars
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

            # ### performance related settings
            buffer_size = 4

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
                        noise_model=noise_model,
                        train_a=train_loc,
                        train_b=train_scale,
                        compute_fim=provide_fim,
                        compute_hessian=provide_hessian,
                        dtype=dtype
                    )
                else:
                    self.batched_data_model = None

            with tf.name_scope("full_data"):
                logger.debug("building full data model")
                # ### alternative definitions for custom observations:
                sample_selection = tf.compat.v1.placeholder_with_default(
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
                    noise_model=noise_model,
                    train_a=train_loc,
                    train_b=train_scale,
                    compute_fim=provide_fim,
                    compute_hessian=provide_hessian,
                    dtype=dtype
                )

            logger.debug("building trainers")
            self._run_trainer_init(
                provide_optimizers=provide_optimizers,
                train_loc=train_loc,
                train_scale=train_scale,
                dtype=dtype
            )

            # Define output metrics:
            logger.debug("building outputs")
            self._set_out_var(
                feature_isnonzero=feature_isnonzero,
                dtype=dtype
            )
            self.loss = self.full_data_model.loss_final
            self.log_likelihood = self.full_data_model.log_likelihood_final
            self.hessian = self.full_data_model.hessians_final
            self.fisher_inv = tf.linalg.inv(-self.full_data_model.hessians_final)  # TODO switch for fim?
            # Summary statistics on feature-wise model gradients:
            self.gradients = tf.reduce_sum(tf.abs(self.full_data_model.neg_jac_final / num_observations), axis=1)

        with tf.name_scope('summaries'):
            if extended_summary:
                tf.summary.histogram('a_var', self.model_vars.a_var)
                tf.summary.histogram('b_var', self.model_vars.b_var)
                tf.summary.scalar('loss', self.full_data_model.loss)
                tf.summary.scalar('learning_rate', self.learning_rate)

        self.saver = tf.compat.v1.train.Saver()
        self.merged_summary = tf.compat.v1.summary.merge_all()
