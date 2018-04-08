import tensorflow as tf

def mixture(weights: tf.Tensor, weight_update: tf.Tensor, probs: tf.Tensor, param_update: tf.Tensor):
    with tf.name_scope("mixture"):
        weight = tf.Variable(weights, name="weight")
        weight_update = tf.assign(weight, weight_update, name="weight_update")
        param_update = tf.assign(probs, param_update, name="param_update")