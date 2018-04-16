import tensorflow as tf
import numpy as np

from tensorflow.contrib import eager as tfe
from tensorflow.python import debug as tf_debug

tfe.enable_eager_execution()

import impl.tf.negative_binomial.util as nb_utils
import impl.tf.util as tf_utils
from models.negative_binomial_mixture_linear_biased.base import validate_data
from models.negative_binomial_mixture_linear_biased import Simulator

sim = Simulator()
# sim.generate()
# sim.save("resources/")

sim.load("resources/")

input_data = validate_data(sim.data)

num_mixtures = input_data["initial_mixture_probs"].shape[0]
(num_samples, num_genes) = input_data["sample_data"].shape

sample_data = tf.convert_to_tensor(sim.data.sample_data, tf.float32)
design = tf.convert_to_tensor(sim.data.design, tf.float32)

###########################
optimizable_nb = False
use_em = True
random_effect = 0.1
learning_rate = 0.05

sample_data = tf.convert_to_tensor(input_data.sample_data, dtype=tf.float32)
design = tf.convert_to_tensor(input_data.design, dtype=tf.float32)
initial_mixture_probs = tf.convert_to_tensor(input_data.initial_mixture_probs, dtype=tf.float32)

# data preparation
with tf.name_scope("prepare_data"):
    # apply a random intercept to avoid zero gradients and infinite values
    with tf.name_scope("randomize"):
        initial_mixture_probs += tf.random_uniform(initial_mixture_probs.shape, 0, random_effect,
                                                   dtype=tf.float32)
        initial_mixture_probs = initial_mixture_probs / tf.reduce_sum(initial_mixture_probs, axis=0,
                                                                      keepdims=True)
        initial_mixture_probs = tf.expand_dims(initial_mixture_probs, -1)
        initial_mixture_probs = tf.identity(initial_mixture_probs, name="adjusted_initial_mixture_probs")
    assert (initial_mixture_probs.shape == (num_mixtures, num_samples, 1))
    
    # broadcast sample data to shape (num_mixtures, num_samples, num_genes)
    with tf.name_scope("broadcast"):
        sample_data = tf.expand_dims(sample_data, axis=0)
        sample_data = tf.tile(sample_data, (num_mixtures, 1, 1))
        # TODO: change tf.tile to tf.broadcast_to after next TF release
    assert (sample_data.shape == (num_mixtures, num_samples, num_genes))

# define mixture_prob tensor depending on optimization method
mixture_prob = initial_mixture_probs

with tf.name_scope("partitions"):
    _, partitions = tf.py_func(
        lambda x: np.unique(x, axis=0, return_inverse=True),
        [design],
        [design.dtype, tf.int64]
    )
    partitions = tf.cast(partitions, tf.int32)

# calculate mixture model probability:
distribution = nb_utils.fit_partitioned(sample_data=sample_data, partitions=partitions,
                                        axis=-2,
                                        weights=mixture_prob,
                                        name="fit_mixture_nb-dist")

count_probs = distribution.prob(sample_data, name="count_probs")
log_count_probs = tf.log(count_probs, name="log_count_probs")
# sum up: for k in num_mixtures: mixture_prob(k) * P(r_k, mu_k, sample_data)
joint_probs = tf_utils.reduce_weighted_mean(count_probs, weight=mixture_prob, axis=-3,
                                            name="joint_probs")

log_probs = tf.log(joint_probs, name="log_probs")

# define training operations
with tf.name_scope("training"):
    # minimize negative log probability (log(1) = 0)
    loss = -tf.reduce_sum(log_probs, name="loss")
    
    # define train function
    em_op = None
    if use_em:
        with tf.name_scope("expectation_maximization"):
            r"""
            E(p_{j,k}) = \frac{P_{x}(j,k)}{\sum_{a}{P_{x}(j,a)}} \\
            P_{x}(j,k) = \prod_{i}{L_{NB}(x_{i,j,k} | \mu_{j,k}, \phi_{j,k})} \\
            log_{P_x}(j, k) = \sum_{i}{log(L_{NB}(x_{i,j,k} | \mu_{j,k}, \phi_{j,k}))} \\
            E(p_{j,k}) = exp(\frac{log_{P_{x}(j,k)}}{log(\sum_{a}{exp(log_{P_{x}}(j,a)}))})

            Here, the log(sum(exp(a))) trick can be used for the denominator to avoid numeric instabilities.
            """
            sum_of_logs = tf.reduce_sum(log_count_probs, axis=-1, keepdims=True)
            expected_weight = sum_of_logs - tf.reduce_logsumexp(sum_of_logs, axis=0, keepdims=True)
            expected_weight = tf.exp(expected_weight)
            expected_weight = tf.identity(expected_weight, name="normalize")
            
            # mixture_prob = expected_weight
    train_op = None
    if optimizable_nb or not use_em:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        if use_em:
            train_op = tf.group(train_op, em_op)
    else:
        train_op = em_op

# parameters
with tf.name_scope("mu"):
    mu = tf.reduce_sum(distribution.mu * mixture_prob, axis=-3)
with tf.name_scope("r"):
    r = tf.reduce_sum(distribution.r * mixture_prob, axis=-3)
with tf.name_scope("p"):
    p = tf.reduce_sum(distribution.p * mixture_prob, axis=-3)
log_mu = tf.log(mu, name="log_mu")
log_r = tf.log(r, name="log_r")
log_p = tf.log(p, name="log_p")

#################################
sess = tf.InteractiveSession()

feed_dict = tf_utils.input_to_feed_dict(tf.get_default_graph(), input_data)

sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)
# sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6064')
# sess.run(optimizer.compute_gradients(loss), feed_dict=feed_dict)

errors = []
for i in range(5):
    (loss_res, train_res) = sess.run((loss, train_op), feed_dict=feed_dict)
    errors.append(loss_res)
    print(i)

(real_r, real_mu) = sess.run((distribution.r, distribution.mu), feed_dict=feed_dict)
real_mixture_prob = sess.run(mixture_prob, feed_dict=feed_dict)

sess.run(log_probs, feed_dict=feed_dict)
sess.run(distribution.prob(sample_data), feed_dict=feed_dict)
sess.run(mixture_prob, feed_dict=feed_dict)

print(sim.r[:, 0])
print(real_r[:, 0])
