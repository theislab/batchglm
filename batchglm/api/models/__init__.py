from . import numpy
try:
    import tensorflow as tf
    if tf.__version__.split(".")[0] == "1":
        from . import tf1
    else:
        tf1 = None
    if tf.__version__.split(".")[0] == "2":
        from . import tf2
    else:
        tf2 = None
except ImportError:
    tf1 = None
    tf2 = None
