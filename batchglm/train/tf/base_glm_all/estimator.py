import abc
from typing import Union
import logging
from enum import Enum

import tensorflow as tf

import numpy as np

from .estimator_graph import EstimatorGraphAll
from .external import MonitoredTFEstimator, InputData, SparseXArrayDataArray


class EstimatorAll(MonitoredTFEstimator, metaclass=abc.ABCMeta):
    """
    Estimator for Generalized Linear Models (GLMs).
    """

    class TrainingStrategy(Enum):
        pass

    model: EstimatorGraphAll
    _train_loc: bool
    _train_scale: bool

    def __init__(
            self,
            input_data: InputData,
            batch_size: int,
            graph: tf.Graph,
            init_a: Union[np.ndarray],
            init_b: Union[np.ndarray],
            model: EstimatorGraphAll,
            provide_optimizers: dict,
            provide_batched: bool,
            provide_fim: bool,
            provide_hessian: bool,
            extended_summary,
            noise_model: str,
            dtype: str
    ):
        """
        Create a new estimator for a GLM-like model.

        :param input_data: InputData
            The input data
        :param batch_size: int
            Size of mini-batches used.
        :param graph: (optional) tf.Graph
        :param init_model: (optional)
            If provided, this model will be used to initialize this Estimator.
        :param init_a: np.ndarray
            Initialization of 'a' (location) model.
        :param init_b: np.ndarray
            Initialization of 'b' (scale) model.
        :param quick_scale: bool
            Whether `scale` will be fitted faster and maybe less accurate.
        :param model: EstimatorGraph
            EstimatorGraph to use. Basically for debugging.
        :param provide_optimizers:

            E.g.    {"gd": False, "adam": False, "adagrad": False, "rmsprop": False,
                    "nr": False, "nr_tr": True, "irls": False, "irls_tr": False}
        :param provide_batched: bool
            Whether mini-batched optimizers should be provided.
        :param extended_summary: Include detailed information in the summaries.
            Will increase runtime of summary writer, use only for debugging.
        :param dtype: Precision used in tensorflow.
        """
        if noise_model == "nb":
            from .external_nb import EstimatorGraph
        elif noise_model == "norm":
            from .external_norm import EstimatorGraph
        elif noise_model == "beta":
            from .external_beta import EstimatorGraph
        else:
            raise ValueError("noise model %s was not recognized" % noise_model)
        self.noise_model = noise_model

        # validate design matrix:
        if np.linalg.matrix_rank(input_data.design_loc) != np.linalg.matrix_rank(input_data.design_loc.T):
            raise ValueError("design_loc matrix is not full rank")
        if np.linalg.matrix_rank(input_data.design_scale) != np.linalg.matrix_rank(input_data.design_scale.T):
            raise ValueError("design_scale matrix is not full rank")

        # ### initialization
        if model is None:
            if graph is None:
                graph = tf.Graph()

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

            if isinstance(input_data.X, SparseXArrayDataArray):
                X_tensor_idx, X_tensor_val, X_shape = tf.py_func(  #tf.py_function( TODO: replace with tf>=v1.13
                    func=input_data.fetch_X_sparse,
                    inp=[idx],
                    Tout=[np.int64, np.float64, np.int64],
                    stateful=False  #  TODO: remove with tf>=v1.13
                )
                # Note on Tout: np.float64 for val seems to be required to avoid crashing v1.12.
                X_tensor_idx = tf.cast(X_tensor_idx, dtype=tf.int64)
                X_shape = tf.cast(X_shape, dtype=tf.int64)
                X_tensor_val = tf.cast(X_tensor_val, dtype=dtype)
                X_tensor = (X_tensor_idx, X_tensor_val, X_shape)
            else:
                X_tensor = tf.py_func(  #tf.py_function( TODO: replace with tf>=v1.13
                    func=input_data.fetch_X_dense,
                    inp=[idx],
                    Tout=input_data.X.dtype,
                    stateful=False  #  TODO: remove with tf>=v1.13
                )
                X_tensor.set_shape(idx.get_shape().as_list() + [input_data.num_features])
                X_tensor = (tf.cast(X_tensor, dtype=dtype),)

            design_loc_tensor = tf.py_func(  #tf.py_function( TODO: replace with tf>=v1.13
                func=input_data.fetch_design_loc,
                inp=[idx],
                Tout=input_data.design_loc.dtype,
                stateful=False  #  TODO: remove with tf>=v1.13
            )
            design_loc_tensor.set_shape(idx.get_shape().as_list() + [input_data.num_design_loc_params])
            design_loc_tensor = tf.cast(design_loc_tensor, dtype=dtype)

            design_scale_tensor = tf.py_func(  #tf.py_function( TODO: replace with tf>=v1.13
                func=input_data.fetch_design_scale,
                inp=[idx],
                Tout=input_data.design_scale.dtype,
                stateful=False  #  TODO: remove with tf>=v1.13
            )
            design_scale_tensor.set_shape(idx.get_shape().as_list() + [input_data.num_design_scale_params])
            design_scale_tensor = tf.cast(design_scale_tensor, dtype=dtype)

            if input_data.size_factors is not None and noise_model in ["nb", "norm"]:
                size_factors_tensor = tf.py_func(  #tf.py_function( TODO: replace with tf>=v1.13
                    func=input_data.fetch_size_factors,
                    inp=[idx],
                    Tout=input_data.size_factors.dtype,
                    stateful=False  #  TODO: remove with tf>=v1.13
                )
                size_factors_tensor.set_shape(idx.get_shape())
                size_factors_tensor = tf.expand_dims(size_factors_tensor, axis=-1)
                size_factors_tensor = tf.cast(size_factors_tensor, dtype=dtype)
            else:
                size_factors_tensor = tf.constant(1, shape=[1, 1], dtype=dtype)

            size_factors_tensor = tf.broadcast_to(size_factors_tensor,
                                                  shape=[tf.size(idx), input_data.num_features])

            # return idx, data
            return idx, (X_tensor, design_loc_tensor, design_scale_tensor, size_factors_tensor)

        with graph.as_default():
            # create model
            model = EstimatorGraph(
                fetch_fn=fetch_fn,
                feature_isnonzero=self.input_data.feature_isnonzero,
                num_observations=self.input_data.num_observations,
                num_features=self.input_data.num_features,
                num_design_loc_params=self.input_data.num_design_loc_params,
                num_design_scale_params=self.input_data.num_design_scale_params,
                num_loc_params=self.input_data.num_loc_params,
                num_scale_params=self.input_data.num_scale_params,
                batch_size=batch_size,
                graph=graph,
                init_a=init_a,
                init_b=init_b,
                constraints_loc=self.input_data.constraints_loc,
                constraints_scale=self.input_data.constraints_scale,
                provide_optimizers=provide_optimizers,
                provide_batched=provide_batched,
                provide_fim=provide_fim,
                provide_hessian=provide_hessian,
                train_loc=self._train_loc,
                train_scale=self._train_scale,
                extended_summary=extended_summary,
                noise_model=self.noise_model,
                dtype=dtype
            )

        MonitoredTFEstimator.__init__(self, model)
        model.session = self.session

    def _scaffold(self):
        with self.model.graph.as_default():
            scaffold = tf.train.Scaffold(
                init_op=self.model.init_op,
                summary_op=self.model.merged_summary,
                saver=self.model.saver,
            )
        return scaffold

    def train(
            self,
            *args,
            learning_rate=None,
            convergence_criteria="all_converged",
            stopping_criteria=None,
            train_loc: bool = None,
            train_scale: bool = None,
            use_batching=False,
            optim_algo=None,
            **kwargs
    ):
        r"""
        Starts training of the model

        :param feed_dict: dict of values which will be feeded each `session.run()`

            See also feed_dict parameter of `session.run()`.
        :param learning_rate: learning rate used for optimization
        :param convergence_criteria: criteria after which the training will be interrupted.
            Currently implemented criterias:

            - "step":
              stop, when the step counter reaches `stopping_criteria`
        :param stopping_criteria: Additional parameter for convergence criteria.

            See parameter `convergence_criteria` for exact meaning
        :param train_loc: Set to True/False in order to enable/disable training of loc
        :param train_scale: Set to True/False in order to enable/disable training of scale
        :param use_batching: If True, will use mini-batches with the batch size defined in the constructor.
            Otherwise, the gradient of the full dataset will be used.
        :param optim_algo: name of the requested train op.
            See :func:train_utils.MultiTrainer.train_op_by_name for further details.
        """
        if train_loc is None:
            # check if mu was initialized with MLE
            train_loc = self._train_loc
        if train_scale is None:
            # check if r was initialized with MLE
            train_scale = self._train_scale

        # Check whether newton-rhapson is desired:
        require_hessian = False
        require_fim = False
        trustregion_mode = False

        if optim_algo.lower() == "newton" or \
                optim_algo.lower() == "newton_raphson" or \
                optim_algo.lower() == "nr":
            require_hessian = True

        if optim_algo.lower() == "irls" or \
                optim_algo.lower() == "iwls" or \
                optim_algo.lower() == "irls_gd" or \
                optim_algo.lower() == "iwls_gd":
            require_fim = True

        if optim_algo.lower() == "newton_tr" or \
                optim_algo.lower() == "nr_tr":
            require_hessian = True
            trustregion_mode = True

        if optim_algo.lower() == "irls_tr" or \
                optim_algo.lower() == "iwls_tr" or \
                optim_algo.lower() == "irls_gd_tr" or \
                optim_algo.lower() == "iwls_gd_tr":
            require_fim = True
            trustregion_mode = True

        # Set learning rate defaults if not set by user.
        if learning_rate is None:
            if require_hessian or require_fim:
                learning_rate = 1
            else:
                learning_rate = 0.5

        # Check that newton-rhapson is called properly:
        if require_hessian or require_fim:
            if learning_rate != 1:
                logging.getLogger("batchglm").warning(
                    "Newton-rhapson or IRLS in base_glm_all is used with learning rate " +
                    str(learning_rate) +
                    ". Newton-rhapson and IRLS should only be used with learning rate = 1."
                )

        # Report all parameters after all defaults were imputed in settings:
        logging.getLogger("batchglm").debug("Optimizer settings in base_glm_all Estimator.train():")
        logging.getLogger("batchglm").debug("learning_rate " + str(learning_rate))
        logging.getLogger("batchglm").debug("convergence_criteria " + str(convergence_criteria))
        logging.getLogger("batchglm").debug("stopping_criteria " + str(stopping_criteria))
        logging.getLogger("batchglm").debug("train_loc " + str(train_loc))
        logging.getLogger("batchglm").debug("train_scale " + str(train_scale))
        logging.getLogger("batchglm").debug("use_batching " + str(use_batching))
        logging.getLogger("batchglm").debug("optim_algo " + str(optim_algo))
        if len(kwargs) > 0:
            logging.getLogger("batchglm").debug("**kwargs: ")
            logging.getLogger("batchglm").debug(kwargs)

        if train_loc or train_scale:
            if use_batching:
                train_op = self.model.trainer_batch.train_op_by_name(optim_algo)
            else:
                train_op = self.model.trainer_full.train_op_by_name(optim_algo)

            super()._train(
                *args,
                feed_dict={"learning_rate:0": learning_rate},
                convergence_criteria=convergence_criteria,
                stopping_criteria=stopping_criteria,
                train_op=train_op,
                trustregion_mode=trustregion_mode,
                require_hessian=require_hessian,
                require_fim=require_fim,
                is_batched=use_batching,
                **kwargs
            )

    @property
    def input_data(self):
        return self._input_data

    @property
    def a_var(self):
        return self.to_xarray("a_var", coords=self.input_data.data.coords)

    @property
    def b_var(self):
        return self.to_xarray("b_var", coords=self.input_data.data.coords)

    @property
    def loss(self):
        return self.to_xarray("loss")

    @property
    def log_likelihood(self):
        return self.to_xarray("log_likelihood", coords=self.input_data.data.coords)

    @property
    def gradients(self):
        return self.to_xarray("gradients", coords=self.input_data.data.coords)

    @property
    def hessians(self):
        return self.to_xarray("hessians", coords=self.input_data.data.coords)

    @property
    def fisher_inv(self):
        return self.to_xarray("fisher_inv", coords=self.input_data.data.coords)

    def finalize(self):
        if self.noise_model == "nb":
            from .external_nb import EstimatorStoreXArray
        elif self.noise_model == "norm":
            from .external_norm import EstimatorStoreXArray
        elif self.noise_model == "beta":
            from .external_beta import EstimatorStoreXArray
        else:
            raise ValueError("noise model not recognized")

        self.session.run(self.model.full_data_model.final_set)
        store = EstimatorStoreXArray(self)
        logging.getLogger("batchglm").debug("Closing session")
        self.close_session()
        return store

    @abc.abstractmethod
    def init_par(
            self,
            input_data,
            init_a,
            init_b,
            init_model
    ):
        pass
