import abc
from typing import Union
import logging
import pprint
from enum import Enum

import tensorflow as tf

import numpy as np

try:
    import anndata
except ImportError:
    anndata = None

from .base import ESTIMATOR_PARAMS
from .base import np_clip_param

from .external import AbstractEstimator, XArrayEstimatorStore, InputData, Model, MonitoredTFEstimator
from .external import data_utils, nb_glm_utils
from .estimator_graph import EstimatorGraph

logger = logging.getLogger(__name__)


class Estimator(AbstractEstimator, MonitoredTFEstimator, metaclass=abc.ABCMeta):
    """
    Estimator for Generalized Linear Models (GLMs) with negative binomial noise.
    Uses the natural logarithm as linker function.
    """

    class TrainingStrategy(Enum):
        AUTO = None
        DEFAULT = [
            {
                "learning_rate": 0.5,
                "convergence_criteria": "scaled_moving_average",
                "stopping_criteria": 1e-5,
                "loss_window_size": 10,
                "use_batching": False,
                "optim_algo": "ADAM",
            },
            {
                "convergence_criteria": "scaled_moving_average",
                "stopping_criteria": 1e-10,
                "loss_window_size": 10,
                "use_batching": False,
                "optim_algo": "newton",
            },
        ]
        EXACT = [
            {
                "learning_rate": 0.5,
                "convergence_criteria": "scaled_moving_average",
                "stopping_criteria": 1e-5,
                "loss_window_size": 10,
                "use_batching": False,
                "optim_algo": "ADAM",
            },
            {
                "convergence_criteria": "scaled_moving_average",
                "stopping_criteria": 1e-10,
                "loss_window_size": 10,
                "use_batching": False,
                "optim_algo": "newton",
            },
        ]
        QUICK = [
            {
                "learning_rate": 0.5,
                "convergence_criteria": "scaled_moving_average",
                "stopping_criteria": 1e-8,
                "loss_window_size": 10,
                "use_batching": False,
                "optim_algo": "ADAM",
            },
        ]
        BY_GENE_ADAM = [
            {
                "learning_rate": 0.5,
                "convergence_criteria": "all_converged",
                "stopping_criteria": 1e-5,
                "use_batching": False,
                "optim_algo": "ADAM",
            },
        ]
        BY_GENE_NR = [
            {
                "convergence_criteria": "all_converged",
                "stopping_criteria": 1e-5,
                "use_batching": False,
                "optim_algo": "newton",
            },
        ]
        PRE_INITIALIZED = [
            {
                "convergence_criteria": "scaled_moving_average",
                "stopping_criteria": 1e-10,
                "loss_window_size": 10,
                "use_batching": False,
                "optim_algo": "newton",
            },
        ]
        CONSTRAINED = [  # Should not contain newton-rhapson right now.
            {
                "learning_rate": 0.5,
                "convergence_criteria": "scaled_moving_average",
                "stopping_criteria": 1e-10,
                "loss_window_size": 10,
                "use_batching": False,
                "optim_algo": "ADAM",
            },
        ]
        CONTINUOUS = [
            {
                "learning_rate": 0.5,
                "convergence_criteria": "scaled_moving_average",
                "stopping_criteria": 1e-5,
                "loss_window_size": 10,
                "use_batching": False,
                "optim_algo": "ADAM",
            },
            {
                "convergence_criteria": "scaled_moving_average",
                "stopping_criteria": 1e-10,
                "loss_window_size": 10,
                "use_batching": False,
                "optim_algo": "newton",
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
            provide_optimizers: dict = {"gd": True, "adam": True, "adagrad": True, "rmsprop": True, "nr": True},
            convergence_type: str = "global_cost",
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
                        groupwise_means, init_a, rmsd_a = nb_glm_utils.closedform_nb_glm_logmu(
                            X=input_data.X,
                            design_loc=input_data.design_loc,
                            constraints=input_data.constraints_loc,
                            size_factors=size_factors_init,
                            link_fn=lambda mu: np.log(np_clip_param(mu, "mu"))
                        )

                        # train mu, if the closed-form solution is inaccurate
                        self._train_mu = not np.all(rmsd_a == 0)

                        # Temporal fix: train mu if size factors are given as closed form may be different:
                        if input_data.size_factors is not None:
                            self._train_mu = True

                        logger.info("Using closed-form MLE initialization for mean")
                        logger.debug("RMSE of closed-form mean:\n%s", rmsd_a)
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
                        init_a_xr = data_utils.xarray_from_data(init_a, dims=("design_loc_params", "features"))
                        init_a_xr.coords["design_loc_params"] = input_data.design_loc.coords["design_loc_params"]
                        init_mu = np.exp(input_data.design_loc.dot(init_a_xr))

                        groupwise_scales, init_b, rmsd_b = nb_glm_utils.closedform_nb_glm_logphi(
                            X=input_data.X,
                            mu=init_mu,
                            design_scale=input_data.design_scale,
                            constraints=input_data.constraints_scale,
                            size_factors=size_factors_init,
                            groupwise_means=None,  # Could only use groupwise_means from a init if design_loc and design_scale were the same.
                            link_fn=lambda r: np.log(np_clip_param(r, "r"))
                        )

                        logger.info("Using closed-form MME initialization for dispersion")
                        logger.debug("RMSE of closed-form dispersion:\n%s", rmsd_b)
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
                provide_optimizers=provide_optimizers,
                train_mu=self._train_mu,
                train_r=self._train_r,
                termination_type=convergence_type,
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
              learning_rate=None,
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

        # Check whether newton-rhapson is desired:
        newton_rhapson_mode = False
        if optim_algo.lower() == "newton" or \
                    optim_algo.lower() == "newton-raphson" or \
                    optim_algo.lower() == "newton_raphson" or \
                    optim_algo.lower() == "nr":
            newton_rhapson_mode = True
        # Set learning rae defaults if not set by user.
        if learning_rate is None:
            if newton_rhapson_mode:
                learning_rate = 1
            else:
                learning_rate = 0.5

        # Check that newton-rhapson is called properly:
        if newton_rhapson_mode:
            if learning_rate != 1:
                logger.warning("Newton-rhapson in nb_glm is used with learing rate " + str(learning_rate) +
                               ". Newton-rhapson should only be used with learing rate =1.")

        # Report all parameters after all defaults were imputed in settings:
        logger.debug("Optimizer settings in nb_glm Estimator.train():")
        logger.debug("learning_rate " + str(learning_rate))
        logger.debug("convergence_criteria " + str(convergence_criteria))
        logger.debug("loss_window_size " + str(loss_window_size))
        logger.debug("stopping_criteria " + str(stopping_criteria))
        logger.debug("train_mu " + str(train_mu))
        logger.debug("train_r " + str(train_r))
        logger.debug("use_batching " + str(use_batching))
        logger.debug("optim_algo " + str(optim_algo))
        if len(kwargs) > 0:
            logger.debug("**kwargs: ")
            logger.debug(kwargs)

        if train_mu or train_r:
            if use_batching:
                loss = self.model.loss
                train_op = self.model.trainer_batch.train_op_by_name(optim_algo)
            else:
                loss = self.model.full_loss
                train_op = self.model.trainer_full.train_op_by_name(optim_algo)

            super().train(*args,
                          feed_dict={"learning_rate:0": learning_rate},
                          convergence_criteria=convergence_criteria,
                          loss_window_size=loss_window_size,
                          stopping_criteria=stopping_criteria,
                          loss=loss,
                          train_op=train_op,
                          **kwargs)

    @property
    def input_data(self) -> InputData:
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
