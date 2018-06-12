import abc
import collections.abc
from typing import Dict, Any, Union

import os
import datetime
import threading

import xarray as xr
import tensorflow as tf

from models import BasicEstimator
from .train import StopAtLossHook, TimedRunHook


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

    @property
    @abc.abstractmethod
    def PARAMS(cls) -> dict:
        """
        This method should return a dict of {parameter: (dim0_name, dim1_name, ..)} mappings
        for all parameters of this estimator.
        """
        pass

    def __init__(self, input_data: xr.Dataset, tf_estimator_graph):
        super().__init__(input_data)

        self.model = tf_estimator_graph
        self.session = None

    def initialize(self):
        self.close_session()
        self.feed_dict = {}

        self.session = tf.Session()

    def close_session(self):
        try:
            self.session.close()
            return True
        except:
            return False

    def run(self, tensor):
        return self.session.run(tensor, feed_dict=self.feed_dict)

    def to_xarray(self, params: list):
        # fetch data
        data = self.get(params)

        # get shape of params
        shapes = self.PARAMS

        output = {key: (shapes[key], data[key]) for key in params}
        output = xr.Dataset(output)

        return output

    def get(self, key: Union[str, collections.abc.Iterable]) -> Union[Any, Dict[str, Any]]:
        """
        Returns the values of the tensor(s) specified by key.

        :param key: Either a string or an iterable list/set/tuple/etc. of strings
        :return: Single array if `key` is a string or a dict {k: value} of arrays if `key` is a collection of strings
        """
        if isinstance(key, str):
            return self.run(self.model.__getattribute__(key))
        elif isinstance(key, collections.abc.Iterable):
            d = {s: self.model.__getattribute__(s) for s in key}
            return self.run(d)

    @property
    def global_step(self):
        return self.get("global_step")

    @property
    def loss(self):
        return self.get("loss")

    def __getitem__(self, key):
        """
        See :func:`TFEstimator.get` for reference
        """
        return self.get(key)


class MonitoredTFEstimator(TFEstimator, metaclass=abc.ABCMeta):
    session: tf.train.MonitoredSession
    working_dir: str

    def __init__(self, input_data: xr.Dataset, tf_estimator_graph: TFEstimatorGraph):
        super().__init__(input_data, tf_estimator_graph)

        self.working_dir = None

    def run(self, tensor):
        return self.session._tf_sess().run(tensor, feed_dict=self.feed_dict)

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
                    run_steps=export_steps + 1 if export_steps is not None else None,
                    run_secs=export_secs + 1 if export_secs is not None else None,
                    call_request_tensors={p: self.model.__getattribute__(p) for p in export},
                    call_fn=lambda sess, step, data: threading.Thread(
                        target=self._save_timestep,
                        args=(step, data)
                    ).start(),
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

    def _save_timestep(self, step: int, data: dict):
        """
        Saves one time step. Special method for TimedRunHook
        
        :param step: the current step which should be saved
        :param data: dict {"param" : data} containing the data which should be saved to disk
        """
        # get shape of params
        shapes = self.PARAMS

        # create mapping: {key: (dimensions, data)}
        xarray = {key: (shapes[key], data) for (key, data) in data.items()}

        xarray = xr.Dataset(xarray)
        xarray["global_step"] = (), step
        xarray["time"] = (), datetime.datetime.now()

        xarray.to_netcdf(path=os.path.join(self.working_dir, "estimation-%d.h5" % step))

    def train(self, *args, feed_dict=None, **kwargs):
        """
        Starts training of the model
        
        :param feed_dict: dict of values which will be feeded each `session.run()`
            
            See also feed_dict parameter of `session.run()`.
        :returns last value of `loss`
        """
        # feed_dict = dict() if feed_dict is None else feed_dict.copy()
        loss_res = None
        while not self.session.should_stop():
            train_step, loss_res, _ = self.session.run(
                (self.model.global_step, self.model.loss, self.model.train_op),
                feed_dict=feed_dict
            )

            tf.logging.info("Step: %d\tloss: %f" % (train_step, loss_res))

        return loss_res
