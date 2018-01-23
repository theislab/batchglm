import tensorflow as tf
from tensorflow.contrib import graph_editor as ge


def initialize_all_variables(sess=None, feed_dict=None):
    """Initializes all uninitialized variables in correct order. Initializers
    are only run for uninitialized variables, so it's safe to run this multiple
    times.

    :param sess: session to use. Use default session if None.
    :param feed_dict: dict to feed
    """

    def make_initializer(var):
        def f():
            return tf.assign(var, var.initial_value).op

        return f

    def make_noop():
        return tf.no_op()

    def make_safe_initializer(var):
        """Returns initializer op that only runs for uninitialized ops."""
        return tf.cond(tf.is_variable_initialized(var), make_noop, make_initializer(var),
                       name="safe_init_" + var.op.name).op

    if not sess:
        sess = tf.get_default_session()
    g = tf.get_default_graph()

    safe_initializers = {}
    for v in tf.global_variables():
        safe_initializers[v.op.name] = make_safe_initializer(v)

    # initializers access variable vaue through read-only value cached in
    # <varname>/read, so add control dependency to trigger safe_initializer
    # on read access
    for v in tf.global_variables():
        var_name = v.op.name
        var_cache = g.get_operation_by_name(var_name + "/read")
        ge.reroute.add_control_inputs(var_cache, [safe_initializers[var_name]])

    sess.run(tf.group(*safe_initializers.values()), feed_dict=feed_dict)

    # remove initializer dependencies to avoid slowing down future variable reads
    for v in tf.global_variables():
        var_name = v.op.name
        var_cache = g.get_operation_by_name(var_name + "/read")
        ge.reroute.remove_control_inputs(var_cache, [safe_initializers[var_name]])
