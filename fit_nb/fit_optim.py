import tensorflow as tf
import tensorflow.contrib as tfcontrib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

################################
# Estimate NB distribution parameters by parameter optimization
################################

# load sample data
x = np.loadtxt("sample_data.tsv", delimiter="\t")
df = pd.read_csv("sample_params.tsv", sep="\t")

# previously sampled data
sample_data = tf.placeholder(tf.float32)

# sample_data.shape: (N,M)
N = tf.shape(sample_data)[0]
N = tf.to_float(N)

# distribution parameters which should be optimized
r_estim = tf.Variable(np.repeat(10.0, 10000), dtype=tf.float32, name="r")
p_estim = tf.Variable(np.repeat(0.5, 10000), dtype=tf.float32, name="p")
# Alternative: closed-form solution for p:
p_estim = N * r_estim / (N * r_estim + tf.reduce_sum(sample_data, axis=0))


distribution = tfcontrib.distributions.NegativeBinomial(total_count=r_estim,
                                                        probs=p_estim,
                                                        name="nb-dist")
probs = distribution.log_prob(sample_data)

# minimize negative log probability (log(1) = 0)
loss = -tf.reduce_sum(probs, name="loss")

train_op = tf.train.AdamOptimizer(learning_rate=0.005)
train_op = train_op.minimize(loss, global_step=tf.train.get_global_step())

errors = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # a = sess.run(tf.reduce_sum(probs, axis=0), feed_dict={sample_data: x})
    for i in range(50):
        (probs_res, loss_res, p_estim_res, r_estim_res, _) = \
            sess.run((probs, loss, p_estim, r_estim, train_op), feed_dict={sample_data: x})
        errors.append(loss_res)
        print(i)

print(np.nanmean(np.abs(r_estim_res - df.r) / np.fmax(r_estim_res, df.r)))
