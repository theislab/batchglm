from . import numpy
try:
    import tensorflow as tf
    from . import tf1
except:
    tf1 = None
