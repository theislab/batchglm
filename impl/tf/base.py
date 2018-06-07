import abc
import collections.abc
from typing import Dict, Any, Union

import xarray as xr
import tensorflow as tf

from models import BasicEstimator
from .util import input_to_feed_dict


class TFEstimatorGraph(metaclass=abc.ABCMeta):
    graph: tf.Graph
    loss: tf.Tensor
    
    def __init__(self, graph=None):
        if graph is None:
            graph = tf.Graph()
        self.graph = graph
    
    @abc.abstractmethod
    def initialize(self, session: tf.Session, feed_dict: dict, *args, **kwargs):
        raise NotImplementedError
    
    @abc.abstractmethod
    def train(self, session: tf.Session, feed_dict: dict, *args, **kwargs):
        raise NotImplementedError
    
    def input_to_feed_dict(self, input_data: xr.Dataset, *args, **kwargs) -> \
            Dict[Union[Union[tf.Tensor, tf.Operation], Any], Any]:
        return input_to_feed_dict(self.graph, input_data)


class TFSession:
    session: tf.Session
    feed_dict: Dict[Union[Union[tf.Tensor, tf.Operation], Any], Any]
    
    def run(self, tensor):
        return self.session.run(tensor, feed_dict=self.feed_dict)


class TFEstimator(BasicEstimator, TFSession):
    model: TFEstimatorGraph
    
    def __init__(self, input_data: xr.Dataset, tf_estimator_graph: TFEstimatorGraph):
        super().__init__(input_data)
        self.model = tf_estimator_graph
        if self.model is not None:
            self.create_new_session()
    
    def close_session(self):
        try:
            self.session.close()
            return True
        except:
            return False
    
    def create_new_session(self) -> None:
        self.close_session()
        with self.model.graph.as_default():
            self.session = tf.Session(graph=self.model.graph)
            self.feed_dict = self.model.input_to_feed_dict(self.input_data)
    
    def initialize(self) -> None:
        self.model.initialize(self.session, self.feed_dict)
    
    def train(self, steps: int, *args, **kwargs) -> None:
        self.model.train(self.session, self.feed_dict, steps=steps, *args, **kwargs)
    
    def evaluate(self, s):
        pass
    
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
    
    def __getitem__(self, key):
        """
        See :func:`TFEstimator.get` for reference
        """
        return self.get(key)
