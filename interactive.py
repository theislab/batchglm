import tensorflow as tf
import numpy as np

from tensorflow.contrib import eager as tfe
from tensorflow.python import debug as tf_debug

# tfe.enable_eager_execution()

import impl.tf.negative_binomial.util as nb_utils
import impl.tf.util as base_utils
from models.negative_binomial_mixture import Simulator

sim = Simulator()
# sim.generate()
# sim.save("resources/")

sim.load("resources/")

input_data = sim.data

num_mixtures = input_data["initial_mixture_probs"].shape[0]
(num_samples, num_distributions) = input_data["sample_data"].shape

###########################

sample_data = tf.placeholder(tf.float32, shape=(num_samples, num_distributions), name="sample_data")
initial_mixture_probs = tf.placeholder(tf.float32,
                                       shape=(num_mixtures, num_samples),
                                       name="initial_mixture_probs")

with tf.name_scope("prepare_data"):
    # apply a random intercept to avoid zero gradients and infinite values
    with tf.name_scope("randomize"):
        initial_mixture_probs += tf.random_uniform(initial_mixture_probs.shape, 0, 0.1,
                                                   dtype=tf.float32)
        initial_mixture_probs = initial_mixture_probs / tf.reduce_sum(initial_mixture_probs, axis=0, keepdims=True)
        initial_mixture_probs = tf.identity(initial_mixture_probs, name="adjusted_initial_mixture_probs")
    
    with tf.name_scope("broadcast"):
        sample_data = tf.expand_dims(sample_data, axis=0)
        sample_data = tf.tile(sample_data, (num_mixtures, 1, 1))

with tf.name_scope("mixture_prob"):
    # optimize logits to keep `mixture_prob` between the interval [0, 1]
    logit_mixture_prob = tf.Variable(base_utils.logit(initial_mixture_probs),
                                     name="logit_prob",
                                     validate_shape=False)
    mixture_prob = tf.sigmoid(logit_mixture_prob, name="prob")
    # normalize: `sum(mixture_prob of one sample) = 1` since the assignment probabilities should sum up to 1
    mixture_prob = tf.identity(mixture_prob / tf.reduce_sum(mixture_prob, axis=0, keepdims=True), name="normalize")
    mixture_prob = tf.expand_dims(mixture_prob, axis=-1)

distribution = nb_utils.fit(sample_data=sample_data,
                            axis=-2,
                            weights=mixture_prob,
                            name="fit_nb-dist")

with tf.name_scope("count_probs"):
    with tf.name_scope("probs"):
        probs = distribution.prob(sample_data)
        # sum up: for k in num_mixtures: mixture_prob(k) * P(r_k, mu_k, sample_data)
        probs = base_utils.reduce_weighted_mean(probs, weight=mixture_prob, axis=-3)
    
    log_probs = tf.log(probs, name="log_probs")

with tf.name_scope("training"):
    # minimize negative log probability (log(1) = 0)
    loss = -tf.reduce_sum(log_probs, name="loss")
    
    # define train function
    optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

sess = tf.InteractiveSession()

feed_dict = base_utils.input_to_feed_dict(tf.get_default_graph(), input_data)

sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)
# sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6064')
sess.run(optimizer.compute_gradients(loss), feed_dict=feed_dict)

errors = []
for i in range(5):
    (loss_res, train_res) = sess.run((loss, train_op), feed_dict=feed_dict)
    errors.append(loss_res)
    print(i)

(real_r, real_mu) = sess.run((distribution.r, distribution.mu), feed_dict=feed_dict)
real_mixture_prob = sess.run(mixture_prob, feed_dict=feed_dict)

sess.run(probs, feed_dict=feed_dict)
sess.run(distribution.prob(sample_data), feed_dict=feed_dict)
sess.run(mixture_prob, feed_dict=feed_dict)

print(sim.r[:, 0])
print(real_r[:, 0])
