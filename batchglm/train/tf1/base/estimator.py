import abc
from enum import Enum
import logging
import numpy as np
import pprint
import tensorflow as tf
import time
from typing import Dict, Any, Union, Iterable

from .external import _EstimatorBase, pkg_constants

logger = logging.getLogger("batchglm")


class TFEstimatorGraph(metaclass=abc.ABCMeta):
    graph: tf.Graph
    loss: tf.Tensor
    init_op: tf.Tensor
    train_op: tf.Tensor
    global_step: tf.Tensor

    def __init__(self, graph=None):
        if graph is None:
            graph = tf.Graph()
        self.graph = graph


class _TFEstimator(metaclass=abc.ABCMeta):

    session: tf.compat.v1.Session
    feed_dict: Dict[Union[Union[tf.Tensor, tf.Operation], Any], Any]
    _param_decorators: Dict[str, callable]

    def __init__(
            self
    ):
        self.session = None
        self.feed_dict = {}
        self._param_decorators = dict()

    def initialize(self):
        self.close_session()
        self.feed_dict = {}
        with self.model.graph.as_default():
            # set up session parameters
            self.session = tf.compat.v1.Session(config=pkg_constants.TF_CONFIG_PROTO)
            self.session.run(self._scaffold().init_op, feed_dict=self.feed_dict)

    def close_session(self):
        if self.session is None:
            return False
        try:
            self.session.close()
            return True
        except (tf.errors.OpError, RuntimeError):
            return False

    def run(self, tensor, feed_dict=None):
        if feed_dict is None:
            feed_dict = self.feed_dict

        return self.session.run(tensor, feed_dict=feed_dict)

    @abc.abstractmethod
    def _scaffold(self) -> tf.compat.v1.train.Scaffold:
        """
        Should create a training scaffold for this Estimator's model

        :return: tf1.compat.v1.train.Scaffold object
        """
        pass

    def _get_unsafe(self, key: Union[str, Iterable]) -> Union[Any, Dict[str, Any]]:
        if isinstance(key, str):
            return self.run(self.model.__getattribute__(key))
        elif isinstance(key, Iterable):
            d = {s: self.model.__getattribute__(s) for s in key}
            return self.run(d)

    def get(self, key: Union[str, Iterable]) -> Union[Any, Dict[str, Any]]:
        """
        Returns the values of the tensor(s) specified by key.

        :param key: Either a string or an iterable list/set/tuple/etc. of strings
        :return: Single array if `key` is a string or a dict {k: value} of arrays if `key` is a collection of strings
        """
        if isinstance(key, str):
            if key not in self.param_shapes():
                raise ValueError("Unknown parameter %s" % key)
        elif isinstance(key, Iterable):
            for k in list(key):
                if k not in self.param_shapes():
                    raise ValueError("Unknown parameter %s" % k)
        return self._get_unsafe(key)

    @property
    def global_step(self):
        return self._get_unsafe("global_step")

    @property
    def loss(self):
        return self._get_unsafe("loss")

    def train_sequence(self, training_strategy):
        if isinstance(training_strategy, Enum):
            training_strategy = training_strategy.value
        elif isinstance(training_strategy, str):
            training_strategy = self.TrainingStrategies[training_strategy].value

        if training_strategy is None:
            training_strategy = self.TrainingStrategies.DEFAULT.value

        logger.info("training strategy:\n%s", pprint.pformat(training_strategy))

        for idx, d in enumerate(training_strategy):
            logger.info("Beginning with training sequence #%d", idx + 1)
            self.train(**d)
            logger.info("Training sequence #%d complete", idx + 1)

    def _train(
            self,
            *args,
            learning_rate=None,
            feed_dict=None,
            convergence_criteria="all_converged",
            stopping_criteria=None,
            train_op=None,
            trustregion_mode=False,
            require_hessian=False,
            require_fim=False,
            is_batched=False,
            **kwargs
    ):
        """
        Starts training of the model

        :param feed_dict: dict of values which will be feeded each `session.run()`

            See also feed_dict parameter of `session.run()`.
        :param convergence_criteria: criteria after which the training will be interrupted.

            Currently implemented criterias:

            - "step":
              stop, when the step counter reaches `stopping_criteria`
        :param stopping_criteria: Additional parameter for convergence criteria.

            See parameter `convergence_criteria` for exact meaning
        :param loss_window_size: specifies `N` in `convergence_criteria`.
        :param train_op: uses this training operation if specified
        """
        # Set default values:
        if stopping_criteria is None:
            if convergence_criteria == "step":
                stopping_criteria = 100

        if train_op is None:
            train_op = self.model.train_op

        # Initialize:
        if pkg_constants.EVAL_ON_BATCHED and is_batched:
            _, _ = self.session.run(
                (self.model.batched_data_model.eval_set,
                 self.model.model_vars.convergence_update),
                feed_dict={self.model.model_vars.convergence_status:
                               np.repeat(False, repeats=self.model.model_vars.converged.shape[0])
                           }
            )
            ll_current = self.session.run(self.model.batched_data_model.norm_neg_log_likelihood)
        else:
            # Have to use eval1 here so that correct object is pulled in trust region.
            _, _ = self.session.run(
                (self.model.full_data_model.eval1_set,
                 self.model.model_vars.convergence_update),
                feed_dict={self.model.model_vars.convergence_status:
                               np.repeat(False, repeats=self.model.model_vars.converged.shape[0])
                           }
            )
            ll_current = self.session.run(self.model.full_data_model.norm_neg_log_likelihood_eval1)

        tf.compat.v1.logging.info(
            "Step: 0 loss: %f models converged 0",
            np.sum(ll_current)
        )

        # Set all to convergence status to False, this is need if multiple training strategies are run:
        converged_current = np.repeat(False, repeats=self.model.model_vars.converged.shape[0])
        train_step = 0

        def convergence_decision(convergence_status, step_counter):
            if convergence_criteria == "step":
                return np.any(np.logical_not(convergence_status)) and step_counter < stopping_criteria
            elif convergence_criteria == "all_converged":
                return np.any(np.logical_not(convergence_status))
            else:
                raise ValueError("convergence_criteria %s not recognized." % convergence_criteria)

        while convergence_decision(converged_current, train_step):
            t0 = time.time()
            converged_prev = converged_current.copy()
            ll_prev = ll_current.copy()

            ## Run update.
            t_a = time.time()
            if is_batched:
                _ = self.session.run(self.model.batched_data_model.train_set)
            else:
                _ = self.session.run(self.model.full_data_model.train_set)

            if trustregion_mode:
                t_b = time.time()
                _, x_step = self.session.run(
                    (train_op["train"]["trial_op"],
                     train_op["update"]),
                    feed_dict=feed_dict
                )
                t_c = time.time()
                _ = self.session.run(self.model.full_data_model.eval0_set)
                t_d = time.time()
                train_step, _, features_updated = self.session.run(
                    (self.model.global_step,
                     train_op["train"]["update_op"],
                     self.model.model_vars.updated),
                    feed_dict=feed_dict
                )
                t_e = time.time()
            else:
                t_b = time.time()
                train_step, _, x_step, features_updated = self.session.run(
                    (self.model.global_step,
                     train_op["train"],
                     train_op["update"],
                     self.model.model_vars.updated),
                    feed_dict=feed_dict
                )
                t_c = time.time()

            if pkg_constants.EVAL_ON_BATCHED and is_batched:
                _ = self.session.run(self.model.batched_data_model.eval_set)
                ll_current, jac_train = self.session.run(
                    (self.model.batched_data_model.norm_neg_log_likelihood,
                     self.model.batched_data_model.neg_jac_train_eval)
                )
            else:
                _ = self.session.run(self.model.full_data_model.eval1_set)
                ll_current, jac_train = self.session.run(
                    (self.model.full_data_model.norm_neg_log_likelihood_eval1,
                     self.model.full_data_model.neg_jac_train_eval)
                )
            t_f = time.time()

            if trustregion_mode:
                tf.compat.v1.logging.debug(
                    "### run time break-down: reduce op. %s, trial %s, ll %s, update %s, eval %s",
                    str(np.round(t_b - t_a, 3)),
                    str(np.round(t_c - t_b, 3)),
                    str(np.round(t_d - t_c, 3)),
                    str(np.round(t_e - t_d, 3)),
                    str(np.round(t_f - t_e, 3))
                )
            else:
                tf.compat.v1.logging.debug(
                    "### run time break-down: reduce op. %s, update %s, eval %s",
                    str(np.round(t_b - t_a, 3)),
                    str(np.round(t_c - t_b, 3)),
                    str(np.round(t_f - t_c, 3))
                )

            if len(self.model.full_data_model.idx_train_loc) > 0:
                x_norm_loc = np.sqrt(np.sum(np.square(
                    np.abs(x_step[self.model.model_vars.idx_train_loc, :])
                ), axis=0))
            else:
                x_norm_loc = np.zeros([self.model.model_vars.n_features])

            if len(self.model.full_data_model.idx_train_scale) > 0:
                x_norm_scale = np.sqrt(np.sum(np.square(
                    np.abs(x_step[self.model.model_vars.idx_train_scale, :])
                ), axis=0))
            else:
                x_norm_scale = np.zeros([self.model.model_vars.n_features])

            # Update convergence status of non-converged features:
            # Cost function value improvement:
            ll_converged = (ll_prev - ll_current) / ll_prev < pkg_constants.LLTOL_BY_FEATURE
            if not pkg_constants.EVAL_ON_BATCHED or not is_batched:
                if np.any(ll_current > ll_prev + 1e-12):
                    tf.compat.v1.logging.warning("bad update found: %i bad updates" % np.sum(ll_current > ll_prev + 1e-12))

            converged_current = np.logical_or(
                converged_prev,
                np.logical_and(ll_converged, features_updated)
            )
            converged_f = np.logical_and(
                np.logical_not(converged_prev),
                np.logical_and(ll_converged, features_updated)
            )
            # Gradient norm:
            if pkg_constants.EVAL_ON_BATCHED and is_batched:
                jac_normalization = self.model.batch_size
            else:
                jac_normalization = self.model.num_observations

            if len(self.model.full_data_model.idx_train_loc) > 0:
                idx_jac_loc = np.array([list(self.model.full_data_model.idx_train).index(x)
                                        for x in self.model.full_data_model.idx_train_loc])
                grad_norm_loc = np.sum(np.abs(jac_train[:, idx_jac_loc]), axis=1) / jac_normalization
            else:
                grad_norm_loc = np.zeros([self.model.model_vars.n_features])
            if len(self.model.full_data_model.idx_train_scale) > 0:
                idx_jac_scale = np.array([list(self.model.full_data_model.idx_train).index(x)
                                          for x in self.model.full_data_model.idx_train_scale])
                grad_norm_scale = np.sum(np.abs(jac_train[:, idx_jac_scale]), axis=1) / jac_normalization
            else:
                grad_norm_scale = np.zeros([self.model.model_vars.n_features])
            converged_g = np.logical_and(
                np.logical_not(converged_prev),
                np.logical_and(
                    grad_norm_loc < pkg_constants.GTOL_BY_FEATURE_LOC,
                    grad_norm_scale < pkg_constants.GTOL_BY_FEATURE_SCALE
                )
            )
            converged_current = np.logical_or(
                converged_current,
                np.logical_and(
                    grad_norm_loc < pkg_constants.GTOL_BY_FEATURE_LOC,
                    grad_norm_scale < pkg_constants.GTOL_BY_FEATURE_SCALE
                )
            )
            # Step length:
            converged_x = np.logical_and(
                np.logical_not(converged_prev),
                np.logical_and(
                    x_norm_loc < pkg_constants.XTOL_BY_FEATURE_LOC,
                    x_norm_scale < pkg_constants.XTOL_BY_FEATURE_SCALE
                )
            )
            converged_current = np.logical_or(
                converged_current,
                np.logical_and(
                    x_norm_loc < pkg_constants.XTOL_BY_FEATURE_LOC,
                    x_norm_scale < pkg_constants.XTOL_BY_FEATURE_SCALE
                )
            )
            t1 = time.time()

            self.session.run((self.model.model_vars.convergence_update), feed_dict={
                self.model.model_vars.convergence_status: converged_current
            })
            tf.compat.v1.logging.info(
                "Step: %d loss: %f, converged %i in %s sec., updated %i, {f: %i, g: %i, x: %i}",
                train_step,
                np.sum(ll_current),
                np.sum(converged_current).astype("int32"),
                str(np.round(t1 - t0, 3)),
                np.sum(np.logical_and(np.logical_not(converged_prev), features_updated)).astype("int32"),
                np.sum(converged_f), np.sum(converged_g), np.sum(converged_x)
            )
