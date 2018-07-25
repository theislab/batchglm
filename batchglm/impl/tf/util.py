from typing import Dict, Any, Union

import xarray as xr
import tensorflow as tf


def input_to_feed_dict(graph: tf.Graph, input_data: Union[dict, xr.Dataset]) \
        -> Dict[Union[Union[tf.Tensor, tf.Operation], Any], Any]:
    """
    Converts some input data to a feedable dict for Tensorflow sessions based on the placeholders in a tf.Graph

    :param graph: tf.Graph object
    :param input_data: either xr.Dataset or some dict{"placeholder": data}
    :return: dict{"placeholder:0", data} for all placeholder names in `input_data`
    """
    placeholders = {op.name: op for op in graph.get_operations() if op.type.lower().startswith("placeholder")}

    if isinstance(input_data, xr.Dataset):
        keys = input_data.variables.keys()
    else:
        keys = input_data.keys()
    keys = set(keys).intersection(placeholders.keys())

    retval = {}
    for k in keys:
        retval[graph.get_tensor_by_name(k + ":0")] = input_data[k]

    return retval
