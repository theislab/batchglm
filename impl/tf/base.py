import abc
from typing import Dict, Any, Union

import tensorflow as tf
from tensorflow import Tensor, Operation

from . import BasicEstimator

__all__ = ['TFEstimatorGraph', 'TFEstimator']


class TFEstimatorGraph(metaclass=abc.ABCMeta):
    graph: tf.Graph
    loss: tf.Tensor
    
    def __init__(self, graph=tf.Graph()):
        self.graph = graph
    
    @abc.abstractmethod
    def initialize(self, session: tf.Session, feed_dict: dict, *args, **kwargs):
        raise NotImplementedError
    
    @abc.abstractmethod
    def train(self, session: tf.Session, feed_dict: dict, *args, **kwargs):
        raise NotImplementedError
    
    def input_to_feed_dict(self, input_data: dict, *args, **kwargs) -> Dict[Union[Union[Tensor, Operation], Any], Any]:
        retval = {}
        with self.graph.as_default():
            for (key, value) in input_data.items():
                retval[self.graph.get_tensor_by_name(key + ":0")] = value
        
        return retval


class TFEstimator(BasicEstimator):
    model: TFEstimatorGraph
    session: tf.Session
    feed_dict: Dict[Union[Union[Tensor, Operation], Any], Any]
    
    def __init__(self, input_data: dict, tf_estimator_graph: TFEstimatorGraph):
        super().__init__(input_data)
        self.model = tf_estimator_graph
        self.create_new_session()
    
    def create_new_session(self) -> None:
        with self.model.graph.as_default():
            self.session = tf.Session()
            self.feed_dict = self.model.input_to_feed_dict(self.input_data)
    
    def initialize(self) -> None:
        self.model.initialize(self.session, self.feed_dict)
    
    def train(self, steps: int, *args, **kwargs) -> None:
        self.model.train(self.session, self.feed_dict, steps, *args, **kwargs)
    
    def run(self, tensor):
        return self.session.run(tensor, feed_dict=self.feed_dict)
    
    # # TODO: hässlich; dämliches Maß; nur für einen Param
    # def compare(self, real_values):
    #     print(np.nanmean(
    #         np.abs(self.r - np.array(real_values.r)) /
    #         np.fmax(self.r, np.array(real_values.r))
    #     ))
    
    def evaluate(self, s):
        pass
