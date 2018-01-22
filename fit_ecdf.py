import tensorflow as tf
import tensorflow.contrib as tfcontrib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def ecdf(x):
    """
    Calculates the ECDF of a 1-D tensor
    :param x: 1-D tensor
    :return: ECDF values of x
    """
    
    def argsort(x):
        _, sorted_indices = tf.nn.top_k(x, tf.shape(x)[0])
        return sorted_indices
    
    y, idx, counts = tf.unique_with_counts(x)
    
    sorting_indices = argsort(-y)
    sorting_reverse_indices = tf.gather(sorting_indices, sorting_indices)
    
    counts_cumsum = tf.cumsum(tf.gather(counts, sorting_indices))
    
    ecdf = tf.gather(counts_cumsum, sorting_reverse_indices)
    ecdf = tf.gather(ecdf, idx)
    ecdf = tf.cast(ecdf, x.dtype) / tf.reduce_sum(x)
    
    return ecdf


################################
# Estimate NB distribution parameters by parameter optimization
################################

# load sample data
x = np.loadtxt("sample_data.tsv", delimiter="\t")
df = pd.read_csv("sample_params.tsv", sep="\t")

x = x[:, range(2)]

# previously sampled data
sample_data = tf.placeholder(tf.float32)

# sample_data.shape: (N,M)
N = tf.shape(sample_data)[0]
N = tf.to_float(N)
M = tf.shape(sample_data)[1]

# distribution parameters which should be optimized
r_estim = tf.Variable(np.repeat(10.0, x.shape[0]), dtype=tf.float32, name="r")
# p_estim = tf.Variable(np.repeat(0.5, 10000), dtype=tf.float32, name="p")
# Alternative: closed-form solution for p:
p_estim = N * r_estim / (N * r_estim + tf.reduce_sum(sample_data, axis=0))

distribution = tfcontrib.distributions.NegativeBinomial(total_count=r_estim,
                                                        probs=p_estim,
                                                        name="nb-dist")
# probs = distribution.log_prob(sample_data)
cdf_estim = tf.map_fn(distribution.cdf, sample_data)
cdf_obs = tf.map_fn(ecdf, sample_data)

# minimize negative log probability (log(1) = 0)
loss = tf.reduce_mean(tf.square(cdf_estim - cdf_obs), name="loss")

train_op = tf.train.AdamOptimizer(learning_rate=0.005)
train_op = train_op.minimize(loss, global_step=tf.train.get_global_step())

errors = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # a = sess.run(tf.reduce_sum(probs, axis=0), feed_dict={sample_data: x})
    for i in range(5):
        (loss_res, p_estim_res, r_estim_res, cdf_estim_res, cdf_obs_res, train_op_res) = \
            sess.run((loss, p_estim, r_estim, cdf_estim, cdf_obs, train_op), feed_dict={sample_data: x})
        errors.append(loss_res)
        print(i)

print(np.nanmean(np.abs(r_estim_res - df.r) / np.fmax(r_estim_res, df.r)))
