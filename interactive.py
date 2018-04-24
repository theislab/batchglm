import tensorflow as tf
import numpy as np

# tfe.enable_eager_execution()

import impl.tf.negative_binomial.util as nb_utils
import impl.tf.util as tf_utils
from impl.tf.linear_regression import LinearRegression
from models.negative_binomial_mixture_linear_biased import Simulator

import utils.stats as stat_utils

sim = Simulator()
# sim.generate()
# sim.save("resources/")

sim.load("resources/")

# input_data = sim.data

num_mixtures = sim.data["initial_mixture_probs"].shape[0]
num_design_params = sim.data["design"].shape[-1]
(num_samples, num_genes) = sim.data["sample_data"].shape

optimizable_nb = False
random_effect = 0.1
learning_rate = 0.05

###########################

sample_data = tf.placeholder(tf.float32, shape=(num_samples, num_genes), name="sample_data")
design = tf.placeholder(tf.float32, shape=(num_samples, num_design_params), name="design")
initial_mixture_probs = tf.placeholder(tf.float32,
                                       shape=(num_mixtures, num_samples),
                                       name="initial_mixture_probs")
learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")

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

    # broadcast sample data to shape (num_mixtures, num_samples, num_design_params)
    with tf.name_scope("broadcast"):
        design = tf.expand_dims(design, axis=0)
        design = tf.tile(design, (num_mixtures, 1, 1))
        # TODO: change tf.tile to tf.broadcast_to after next TF release
    assert (design.shape == (num_mixtures, num_samples, num_design_params))

# define mixture_prob tensor depending on optimization method
mixture_prob = None
with tf.name_scope("mixture_prob"):
    # optimize logits to keep `mixture_prob` between the interval [0, 1]
    logit_mixture_prob = tf.Variable(tf_utils.logit(initial_mixture_probs), name="logit_prob")
    mixture_prob = tf.sigmoid(logit_mixture_prob, name="prob")

    # normalize: the assignment probabilities should sum up to 1
    # => `sum(mixture_prob of one sample) = 1`
    mixture_prob = mixture_prob / tf.reduce_sum(mixture_prob, axis=0, keepdims=True)
    mixture_prob = tf.identity(mixture_prob, name="normalize")
assert (mixture_prob.shape == (num_mixtures, num_samples, 1))

with tf.name_scope("initialization"):
    init_dist = nb_utils.fit(sample_data, weights=initial_mixture_probs, axis=-2)

    init_a_intercept = tf.log(init_dist.mean())
    init_a_slopes = tf.truncated_normal([1, num_design_params - 1, num_genes],
                                        dtype=design.dtype)

    init_b_intercept = tf.log(init_dist.variance())
    init_b_slopes = tf.truncated_normal([1, num_design_params - 1, num_genes],
                                        dtype=design.dtype)

# define variables
mu_obs = tf.Variable(tf.tile(init_dist.mean(), (1, num_samples, 1)), name="mu")
sigma2_obs = tf.Variable(tf.tile(init_dist.variance(), (1, num_samples, 1)), name="var")

log_mu_obs = tf.log(mu_obs, name="log_mu_obs")
log_sigma2_obs = tf.log(sigma2_obs, name="log_var_obs")

with tf.name_scope("variable_a"):
    a_intercept = tf.Variable(init_a_intercept, name='a_intercept')
    a_slopes = tf.Variable(init_a_slopes, name='a_slopes')
    a_slopes = tf.tile(a_slopes, (num_mixtures, 1, 1), name="constraint_a")
    a = tf.concat([
        a_intercept,
        a_slopes
    ], axis=-2)
    a = tf.identity(a, name="a")
assert a.shape == (num_mixtures, num_design_params, num_genes)

linreg_a = LinearRegression(design, log_sigma2_obs, b=a, name="lin_reg_a", fast=False)

with tf.name_scope("variable_b"):
    b_intercept = tf.Variable(init_b_intercept, name='a_intercept')
    b_slopes = tf.Variable(init_b_slopes, name='a_slopes')
    b_slopes = tf.tile(b_slopes, (num_mixtures, 1, 1), name="constraint_a")
    b = tf.concat([
        b_intercept,
        b_slopes
    ], axis=-2)
    b = tf.identity(b, name="b")
assert a.shape == (num_mixtures, num_design_params, num_genes)

linreg_b = LinearRegression(design, log_mu_obs, b=b, name="lin_reg_b", fast=False)

# calculate mixture model probability:
distribution = nb_utils.NegativeBinomial(mean=mu_obs, variance=sigma2_obs, name="NB_dist")

count_probs = distribution.prob(sample_data, name="count_probs")
# sum up: for k in num_mixtures: mixture_prob(k) * P(r_k, mu_k, sample_data)
joint_probs = tf_utils.reduce_weighted_mean(count_probs, weight=mixture_prob, axis=-3,
                                            name="joint_probs")

log_probs = tf.log(joint_probs, name="log_probs")

# define training operations
with tf.name_scope("training"):
    # minimize negative log probability (log(1) = 0)
    loss = -tf.reduce_sum(log_probs, name="loss")

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

with tf.name_scope("var_estim"):
    var_estim = tf.exp(tf.gather(a, tf.zeros(num_samples, dtype=tf.int32), axis=-2))
    # var_estim = tf.exp(a[:, 0, :])
with tf.name_scope("mu_estim"):
    mu_estim = tf.exp(tf.gather(b, tf.zeros(num_samples, dtype=tf.int32), axis=-2))
    # mu_estim = tf.exp(b[:, 0, :])
dist_estim = nb_utils.NegativeBinomial(variance=var_estim, mean=mu_estim, name="dist_estim")

mu = tf.reduce_sum(distribution.mean() * mixture_prob, axis=-3)
sigma2 = tf.reduce_sum(distribution.variance() * mixture_prob, axis=-3)
r = tf.reduce_sum(distribution.r * mixture_prob, axis=-3)

#################################
sess = tf.InteractiveSession()

feed_dict = tf_utils.input_to_feed_dict(tf.get_default_graph(), sim.data)

sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)


# sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6064')
# sess.run(optimizer.compute_gradients(loss), feed_dict=feed_dict)

def run(t):
    return sess.run(t, feed_dict=feed_dict)


def train(feed_dict, steps=10, learning_rate=0.05):
    errors = []
    feed_dict = feed_dict.copy()
    feed_dict["learning_rate:0"] = learning_rate
    for i in range(steps):
        (loss_res, train_res) = sess.run((loss, train_op), feed_dict=feed_dict)
        errors.append(loss_res)
        print(i)


for i in range(10):
    print("loss: %d" % sess.run(loss, feed_dict=feed_dict))
    print(stat_utils.mapd(run(sigma2), sim.sigma2))
    print(stat_utils.mapd(run(r), ))
    print(stat_utils.mapd(run(mu), sim.mu))
    print(stat_utils.mae(run(tf.squeeze(mixture_prob)), sim.mixture_prob))

    train(feed_dict, steps=10, learning_rate=0.5)

(real_r, real_mu) = sess.run((distribution.r, distribution.mean()), feed_dict=feed_dict)
real_mixture_prob = sess.run(mixture_prob, feed_dict=feed_dict)

sess.run(log_probs, feed_dict=feed_dict)
sess.run(distribution.prob(sample_data), feed_dict=feed_dict)
sess.run(mixture_prob, feed_dict=feed_dict)

