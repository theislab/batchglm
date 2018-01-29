import tensorflow as tf
import tensorflow.contrib as tfcontrib
import numpy as np

from models import negative_binomial

def buildGraph():
    sample_data = tf.placeholder(tf.float32)

    distribution = negative_binomial.fit(sample_data)
    probs = distribution.log_prob(sample_data)

    # minimize negative log probability (log(1) = 0)
    loss = -tf.reduce_sum(probs, name="loss")

    train_op = tf.train.AdamOptimizer(learning_rate=0.05)
    train_op = train_op.minimize(loss, global_step=tf.train.get_global_step())

    return sample_data, distribution, loss, train_op
