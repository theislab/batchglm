from typing import Union
import logging

import tensorflow as tf
import numpy as np
import xarray as xr

from .external import EstimatorGraphGLM, FullDataModelGraphGLM, BatchedDataModelGraphGLM, ModelVarsGLM
from .external import op_utils
from .external import pkg_constants

logger = logging.getLogger(__name__)


class FullDataModelGraphEval(FullDataModelGraphGLM):
    """
    Basic computational graph to evaluate negative binomial GLM metrics on full data set.

    Evaluate model and cost function and Jacobians.
    """

    def __init__(
            self,
            sample_indices: tf.Tensor,
            fetch_fn,
            batch_size: Union[int, tf.Tensor],
            model_vars,
            constraints_loc,
            constraints_scale,
            noise_model,
            train_a,
            train_b,
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
            from .external_nb import BasicModelGraph, Jacobians
        else:
            raise ValueError("noise model not recognized")
        self.noise_model = noise_model

        data_indices = tf.data.Dataset.from_tensor_slices(sample_indices)
        batched_data = data_indices.batch(batch_size)
        batched_data = batched_data.map(fetch_fn, num_parallel_calls=pkg_constants.TF_NUM_THREADS)

        def map_sparse(idx, data):
            X_tensor_ls, design_loc_tensor, design_scale_tensor, size_factors_tensor = data
            if False: #len(X_tensor_ls) > 1:
                X_tensor = tf.SparseTensor(X_tensor_ls[0], X_tensor_ls[1], X_tensor_ls[2])
            else:
                X_tensor = X_tensor_ls#[0]
            return idx, (X_tensor, design_loc_tensor, design_scale_tensor, size_factors_tensor)

        batched_data = batched_data.map(map_sparse, num_parallel_calls=pkg_constants.TF_NUM_THREADS)
        batched_data = batched_data.prefetch(1)

        with tf.name_scope("log_likelihood"):
            def init_fun():
                return tf.zeros([model_vars.n_features], dtype=dtype)

            def map_fun(idx, data) -> BasicModelGraph:
                X, design_loc, design_scale, size_factors = data
                print(X)
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

            #log_likelihood = batched_data.reduce(
            #    initial_state=init_fun(),
            #    reduce_func=lambda old, new: reduce_fun(old, map_fun(new).log_likelihood)
            #)

            model = map_fun(*fetch_fn(sample_indices))

            with tf.name_scope("log_likelihood"):
                log_likelihood = op_utils.map_reduce(
                    last_elem=tf.gather(sample_indices, tf.size(sample_indices) - 1),
                    data=batched_data,
                    map_fn=lambda idx, data: map_fun(idx, data).log_likelihood,
                    parallel_iterations=1,
                )

            norm_log_likelihood = log_likelihood / tf.cast(tf.size(sample_indices), dtype=log_likelihood.dtype)
            norm_neg_log_likelihood = - norm_log_likelihood
            loss = tf.reduce_sum(norm_neg_log_likelihood)

        # Save attributes necessary for reinitialization:
        self.X = model.X
        self.design_loc = model.design_loc
        self.design_scale = model.design_scale
        self.mu = model.mu
        self.r = model.r
        self.sigma2 = model.sigma2
        self.probs = model.probs
        self.log_probs = model.log_probs

        self.fetch_fn = fetch_fn
        self.batch_size = batch_size
        self.train_a = train_a
        self.train_b = train_b
        self.dtype = dtype
        self.constraints_loc = constraints_loc
        self.constraints_scale = constraints_scale

        self.batched_data = batched_data
        self.sample_indices = sample_indices

        self.log_likelihood = log_likelihood
        self.norm_log_likelihood = norm_log_likelihood
        self.norm_neg_log_likelihood = norm_neg_log_likelihood
        self.loss = loss

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
            if train_a or train_b:
                if not train_a or not train_b:
                    jacobian_train = Jacobians(
                        batched_data=self.batched_data,
                        sample_indices=self.sample_indices,
                        constraints_loc=self.constraints_loc,
                        constraints_scale=self.constraints_scale,
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

        self.jac = jacobian_full.jac
        self.jac_train = jacobian_train


class FullDataModelGraph(FullDataModelGraphEval):
    """
    Extended computational graph to evaluate negative binomial GLM metrics on full data set.

    Inherits base graph which evaluates cost function and adds Hessians and Fisher information matrix.
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
            from .external_nb import Hessians, FIM
        else:
            raise ValueError("noise model not recognized")

        FullDataModelGraphEval.__init__(
            self=self,
            sample_indices=sample_indices,
            fetch_fn=fetch_fn,
            batch_size=batch_size,
            model_vars=model_vars,
            constraints_loc=constraints_loc,
            constraints_scale=constraints_scale,
            train_a=train_a,
            train_b=train_b,
            noise_model=noise_model,
            dtype=dtype
        )

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
            if train_a or train_b:
                if not train_a or not train_b:
                    hessians_train = Hessians(
                        batched_data=self.batched_data,
                        sample_indices=self.sample_indices,
                        constraints_loc=self.constraints_loc,
                        constraints_scale=self.constraints_scale,
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
            # Fisher information matrix of sub-model which is to be trained.
            if train_a or train_b:
                if not train_a or not train_b:
                    fim_train = FIM(
                        batched_data=self.batched_data,
                        sample_indices=self.sample_indices,
                        constraints_loc=self.constraints_loc,
                        constraints_scale=self.constraints_scale,
                        model_vars=model_vars,
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


        self.hessians = hessians_full
        self.hessians_train = hessians_train

        self.fim = fim_full
        self.fim_train = fim_train


class BatchedDataModelGraphEval(BatchedDataModelGraphGLM):
    """
    Basic computational graph to evaluate negative binomial GLM metrics on batched data set.

    Evaluate model and cost function and Jacobians.
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
            from .external_nb import BasicModelGraph, Jacobians
        else:
            raise ValueError("noise model not rewcognized")
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
                if False: #len(X_tensor_ls) > 1:
                    X_tensor = tf.SparseTensor(X_tensor_ls[0], X_tensor_ls[1], X_tensor_ls[2])
                else:
                    X_tensor = X_tensor_ls#[0]
                return idx, (X_tensor, design_loc_tensor, design_scale_tensor, size_factors_tensor)

            #training_data = training_data.map(map_sparse, num_parallel_calls=pkg_constants.TF_NUM_THREADS)

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


class BatchedDataModelGraph(BatchedDataModelGraphEval):
    """
    Extended computational graph to evaluate negative binomial GLM metrics on batched data set

    Inherits base graph which evaluates cost function and adds Hessians and Fisher information matrix.
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
            from .external_nb import BasicModelGraph, Hessians, FIM
        else:
            raise ValueError("noise model not rewcognized")

        BatchedDataModelGraphEval.__init__(
            self=self,
            num_observations=num_observations,
            fetch_fn=fetch_fn,
            batch_size=batch_size,
            buffer_size=buffer_size,
            model_vars=model_vars,
            constraints_loc=constraints_loc,
            constraints_scale=constraints_scale,
            train_a=train_a,
            train_b=train_b,
            noise_model=noise_model,
            dtype=dtype
        )

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
        if train_a or train_b:
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
        self.fim_train = batch_fim


class EstimatorGraphAll(EstimatorGraphGLM):
    """

    Contains model_vars, full_data_model and batched_data_model which are the
    primary training objects. All three also exist as *_eval which can be used
    to perform and iterative optmization within a single parameter update, such
    as during a line search.
    """

    class _FullDataModelGraphEval(FullDataModelGraphEval):
        def __init__(
                self,
                full_data_model: FullDataModelGraph
        ):
            self.sample_indices = full_data_model.sample_indices
            self.fetch_fn = full_data_model.fetch_fn
            self.batch_size = full_data_model.batch_size
            self.constraints_loc = full_data_model.constraints_loc
            self.constraints_scale = full_data_model.constraints_scale
            self.noise_model = full_data_model.noise_model
            self.train_a = full_data_model.train_a
            self.train_b = full_data_model.train_b
            self.dtype = full_data_model.dtype

        def new(
                self,
                model_vars: ModelVarsGLM
        ):
            FullDataModelGraphEval.__init__(
                self=self,
                sample_indices=self.sample_indices,
                fetch_fn=self.fetch_fn,
                batch_size=self.batch_size,
                model_vars=model_vars,
                constraints_loc=self.constraints_loc,
                constraints_scale=self.constraints_scale,
                noise_model=self.noise_model,
                train_a=self.train_a,
                train_b=self.train_b,
                dtype=self.dtype
            )

    class _BatchedDataModelGraphEval(BatchedDataModelGraphEval):
        def __init__(
                self,
                batched_data_model: BatchedDataModelGraph
        ):
            self.num_observations = batched_data_model.num_observations
            self.fetch_fn = batched_data_model.fetch_fn
            self.batch_size = batched_data_model.batch_size
            self.buffer_size = batched_data_model.buffer_size
            self.constraints_loc = batched_data_model.constraints_loc
            self.constraints_scale = batched_data_model.constraints_scale
            self.noise_model = batched_data_model.noise_model
            self.train_a = batched_data_model.train_a
            self.train_b = batched_data_model.train_b
            self.dtype = batched_data_model.dtype

        def new(
                self,
                model_vars: ModelVarsGLM
        ):
            BatchedDataModelGraphEval.__init__(
                self=self,
                num_observations=self.num_observations,
                fetch_fn=self.fetch_fn,
                batch_size=self.batch_size,
                buffer_size=self.buffer_size,
                model_vars=model_vars,
                constraints_loc=self.constraints_loc,
                constraints_scale=self.constraints_scale,
                noise_model=self.noise_model,
                train_a=self.train_a,
                train_b=self.train_b,
                dtype=self.dtype
            )

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
            dtype="float64"
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
            from .external_nb import BasicModelGraph, ModelVars, ModelVarsEval, Jacobians, Hessians, FIM
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
                self.model_vars_eval = ModelVarsEval(
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
