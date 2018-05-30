import xarray as xr
import tensorflow as tf
from typing import Dict, Any, Union


def input_to_feed_dict(graph, input_data: xr.Dataset) -> Dict[Union[Union[tf.Tensor, tf.Operation], Any], Any]:
    """
    converts a
    :param graph:
    :param input_data:
    :return:
    """
    placeholders = {op.name: op for op in graph.get_operations() if op.type.lower().startswith("placeholder")}
    
    keys = set(input_data.variables.keys()).intersection(placeholders.keys())
    
    retval = {}
    for k in keys:
        retval[graph.get_tensor_by_name(k + ":0")] = input_data[k]
    
    return retval


def randomize(tensor, modifier=tf.multiply, min=0.5, max=2.0, name="randomize"):
    with tf.name_scope(name):
        tensor = modifier(tensor, tf.random_uniform(tensor.shape, min, max,
                                                    dtype=tf.float32))
    return tensor
