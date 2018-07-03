import abc
from typing import Dict, Any, Union, List, Iterable

import os
import datetime

import numpy as np
import xarray as xr
import tensorflow as tf

import pkg_constants
from models.base import BasicEstimator
from .train import StopAtLossHook, TimedRunHook
import utils.stats as stat_utils


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


class TFEstimator(BasicEstimator, metaclass=abc.ABCMeta):
    model: TFEstimatorGraph
    session: tf.Session
    feed_dict: Dict[Union[Union[tf.Tensor, tf.Operation], Any], Any]

    def __init__(self, input_data: xr.Dataset, tf_estimator_graph):
        super().__init__(input_data)

        self.model = tf_estimator_graph
        self.session = None

    def initialize(self):
        self.close_session()
        self.feed_dict = {}

        self.session = tf.Session()

    def close_session(self):
        if self.session is None:
            return False
        try:
            self.session.close()
            return True
        except tf.errors.OpError:
            return False

    def run(self, tensor):
        return self.session.run(tensor, feed_dict=self.feed_dict)

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
        return self.get("global_step")

    @property
    def loss(self):
        return self.get("loss")

    def _train_to_convergence(self,
                              feed_dict,
                              loss_history_size,
                              stop_at_loss_change,
                              convergence_criteria="t_test"):

        previous_loss_hist = np.tile(np.inf, loss_history_size)
        loss_hist = np.tile(np.inf, loss_history_size)

        def should_stop(step):
            if step % len(loss_hist) == 0 and not np.any(np.isinf(previous_loss_hist)):
                if convergence_criteria == "simple":
                    change = loss_hist[-2] - loss_hist[-1]
                    tf.logging.info("loss change: %f", change)
                    return change < stop_at_loss_change
                if convergence_criteria == "moving_average":
                    change = np.mean(previous_loss_hist) - np.mean(loss_hist)
                    tf.logging.info("loss change: %f", change)
                    return change < stop_at_loss_change
                elif convergence_criteria == "absolute_moving_average":
                    change = np.abs(np.mean(previous_loss_hist) - np.mean(loss_hist))
                    tf.logging.info("absolute loss change: %f", change)
                    return change < stop_at_loss_change
                elif convergence_criteria == "t_test":
                    pval = stat_utils.welch_t_test(previous_loss_hist, loss_hist)
                    tf.logging.info("pval: %f", pval)
                    return pval > stop_at_loss_change
            else:
                return False

        while True:
            train_step, global_loss, _ = self.session.run(
                (self.model.global_step, self.model.loss, self.model.train_op),
                feed_dict=feed_dict
            )

            tf.logging.info("Step: %d\tloss: %f", train_step, global_loss)

            # update last_loss every N+1st step:
            if train_step % len(loss_hist) == 1:
                previous_loss_hist = np.copy(loss_hist)

            loss_hist[(train_step - 1) % len(loss_hist)] = global_loss

            # check convergence every N steps:
            if should_stop(train_step):
                break

        return np.mean(loss_hist)

    def train(self, *args,
              feed_dict=None,
              convergence_criteria="t_test",
              loss_history_size=None,
              stop_at_loss_change=None,
              **kwargs):
        """
        Starts training of the model

        :param feed_dict: dict of values which will be feeded each `session.run()`

            See also feed_dict parameter of `session.run()`.
        :param convergence_criteria: criteria after which the training will be interrupted.

            Currently implemented criterias:

            "simple":
                stop, when `loss(step=i) - loss(step=i-1)` < `stop_at_loss_change`
            "moving_average":
                stop, when `mean_loss(steps=[i-2N..i-N) - mean_loss(steps=[i-N..i)` < `stop_at_loss_change`
            "absolute_moving_average":
                stop, when `|mean_loss(steps=[i-2N..i-N) - mean_loss(steps=[i-N..i)|` < `stop_at_loss_change`
            "t_test" (recommended):
                Perform t-Test between the last [i-2N..i-N] and [i-N..i] losses.
                Stop if P("both distributions are equal") > `stop_at_loss_change`.
        :param stop_at_loss_change: Additional parameter for convergence criteria.

            See parameter `convergence_criteria` for exact meaning
        :param loss_history_size: specifies `N` in `convergence_criteria`.

        :returns last value of `loss`
        """
        # feed_dict = dict() if feed_dict is None else feed_dict.copy()

        # default values:
        if loss_history_size is None:
            loss_history_size = 200
        if stop_at_loss_change is None:
            if convergence_criteria in ["simple", "moving_agerage", "absolute_moving_average"]:
                stop_at_loss_change = 1e-5
            else:
                stop_at_loss_change = 0.05

        self._train_to_convergence(
            convergence_criteria=convergence_criteria,
            loss_history_size=loss_history_size,
            stop_at_loss_change=stop_at_loss_change,
            feed_dict=feed_dict
        )


class MonitoredTFEstimator(TFEstimator, metaclass=abc.ABCMeta):
    session: tf.train.MonitoredSession
    working_dir: str

    def __init__(self, input_data: xr.Dataset, tf_estimator_graph: TFEstimatorGraph):
        super().__init__(input_data, tf_estimator_graph)

        self.working_dir = None

    def run(self, tensor, feed_dict=None):
        if feed_dict is None:
            feed_dict = self.feed_dict
        return self.session._tf_sess().run(tensor, feed_dict=feed_dict)

    @abc.abstractmethod
    def _scaffold(self) -> tf.train.Scaffold:
        """
        Should create a training scaffold for this Estimator's model
        
        :return: tf.train.Scaffold object
        """
        pass

    def initialize(
            self,
            working_dir: str = None,
            save_checkpoint_steps=None,
            save_checkpoint_secs=None,
            save_summaries_steps=None,
            save_summaries_secs=None,
            stop_at_step=None,
            stop_below_loss_change=None,
            loss_averaging_steps=50,
            export_steps=None,
            export_secs=None,
            export: list = None,
            export_compression=True,
    ):
        """
        Initializes this Estimator.
        
        If specified, previous checkpoints will be loaded from `working_dir`.
    
        :param working_dir: working directory for all actions requiring writing files to disk
        :param save_checkpoint_steps: number of steps after which a new checkpoint will be created
        :param save_checkpoint_secs: period of time after which a new checkpoint will be created
        :param save_summaries_steps: number of steps after which a new summary will be created
        :param save_summaries_secs: period of time after which a new summary will be created
        :param stop_at_step: the step after which the training will be interrupted
        :param stop_below_loss_change: training will be interrupted as soon as the loss improvement drops
            below this value
        :param loss_averaging_steps: if `stop_below_loss_change` is used, this parameter specifies the number of
            steps used to average the loss.

            E.g. a value of '25' would mean that the loss change at step `i` would be calculated as
                `mean_loss(i-24, [...], i) - mean_loss(i-49, [...], i-25)`.
            Useful in cases where the loss is not monotonously falling, e.g. when using mini-batches.
        :param export: list of parameter names.
        
            These parameters will be fetched from `model` and exported as NetCDF4-formatted `xarray.dataset`'s.
            See keys of `estimator.PARAMS` for possible parameters.
        :param export_steps: number of steps after which the parameters specified in `export` will be exported
        :param export_secs: time period after which the parameters specified in `export` will be exported
        :param export_compression: Enable compression for exported data. Defaults to `True`.
        """

        self.close_session()
        self.feed_dict = {}
        self.working_dir = working_dir

        if working_dir is None and not all(val is None for val in [
            save_checkpoint_steps,
            save_checkpoint_secs,
            save_summaries_steps,
            save_summaries_secs,
            export_steps,
            export_secs
        ]):
            raise ValueError("No working_dir provided but actions saving data requested")

        with self.model.graph.as_default():
            # set up session parameters
            scaffold = self._scaffold()

            hooks = [tf.train.NanTensorHook(self.model.loss), ]
            if export_secs is not None or export_steps is not None:
                hooks.append(TimedRunHook(
                    run_steps=export_steps if export_steps is not None else None,
                    run_secs=export_secs if export_secs is not None else None,
                    call_request_tensors={p: self.model.__getattribute__(p) for p in export},
                    call_fn=lambda sess, step, time_measures, data: self._save_timestep(step, time_measures, data),
                    asynchronous=True,
                ))
            if stop_at_step is not None:
                hooks.append(tf.train.StopAtStepHook(last_step=stop_at_step))
            if stop_below_loss_change is not None:
                hooks.append(StopAtLossHook(
                    self.model.loss,
                    min_loss_change=stop_below_loss_change,
                    loss_averaging_steps=loss_averaging_steps
                ))

            # create session
            self.session = tf.train.MonitoredTrainingSession(
                checkpoint_dir=self.working_dir,
                scaffold=scaffold,
                hooks=hooks,
                save_checkpoint_steps=save_checkpoint_steps,
                save_checkpoint_secs=save_checkpoint_secs,
                save_summaries_steps=save_summaries_steps,
                save_summaries_secs=save_summaries_secs,

            )

    def _save_timestep(self, step: int, time_measures: List[float], data: dict, compression=True):
        """
        Saves one time step. Special method for TimedRunHook
        
        :param step: the current step which should be saved
        :param data: dict {"param" : data} containing the data which should be saved to disk
        :param compression: if None, no compression will be used.
            Otherwise the specified compression will be used for all variables.
        """
        # get shape of params
        shapes = self.param_shapes()

        # create mapping: {key: (dimensions, data)}
        xarray = {key: (shapes[key], data) for (key, data) in data.items()}

        xarray = xr.Dataset(xarray)
        xarray.coords["global_step"] = (), step
        xarray.coords["current_time"] = (), datetime.datetime.now()
        xarray.coords["time_elapsed"] = (), (np.sum(time_measures) if len(time_measures) > 0 else 0)

        encoding = None
        if compression:
            opts = dict()
            opts["zlib"] = True

            encoding = {var: opts for var in xarray.data_vars if xarray[var].shape != ()}

        path = os.path.join(self.working_dir, "estimation-%d.h5" % step)
        tf.logging.info("Exporting data to %s" % path)
        xarray.to_netcdf(path=path,
                         engine=pkg_constants.XARRAY_NETCDF_ENGINE,
                         encoding=encoding)
        tf.logging.info("Exporting to %s finished" % path)

    def train(self, *args,
              use_stop_hooks=False,
              **kwargs):
        """
        See TFEstimator.train() for more options

        :param use_stop_hooks: [Experimental]

            If true, session run hooks have to call `request_stop` to end training.

            See `tf.train.SessionRunHook` for details.
        """
        if use_stop_hooks:
            while not self.session.should_stop():
                train_step, loss_res, _ = self.session.run(
                    (self.model.global_step, self.model.loss, self.model.train_op),
                    feed_dict=kwargs.get("feed_dict", None)
                )

                tf.logging.info("Step: %d\tloss: %f" % (train_step, loss_res))
        else:
            super().train(*args, **kwargs)
