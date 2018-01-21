#!/usr/bin/env python3

import tensorflow as tf
import tensorflow.contrib as tfcontrib

import pandas as pd
import numpy as np

from .models import NegativeBinomial

# import matplotlib.pyplot as plt

################################
# Estimate NB distribution parameters using MME
################################


if __name__ == '__main__':
    # load sample data
    x = np.loadtxt("sample_data.tsv", delimiter="\t")
    df = pd.read_csv("sample_params.tsv", sep="\t")
    
    # previously sampled data
    sample_data = tf.placeholder(tf.float32)
    
    # sample_data.shape: (N,M)
    N = tf.shape(sample_data)[0]
    N = tf.to_float(N)
    
    # distribution parameters which should be optimized
    # r_estim = tf.Variable(np.repeat(10.0, 10000), dtype=tf.float32, name="r")
    # p_estim = tf.Variable(np.repeat(0.5, 10000), dtype=tf.float32, name="p")
    # Alternative: closed-form solution for p:
    # p_estim = N * r_estim / (N * r_estim + tf.reduce_sum(sample_data, axis=0))
    
    r, p = NegativeBinomial.fit(sample_data)
    
    distribution = tfcontrib.distributions.NegativeBinomial(total_count=r,
                                                            probs=p,
                                                            name="nb-dist")
    probs = distribution.log_prob(sample_data)
    
    # minimize negative log probability (log(1) = 0)
    loss = -tf.reduce_sum(probs, name="loss")
    
    # train_op = tf.train.AdamOptimizer(learning_rate=0.005)
    # train_op = train_op.minimize(loss, global_step=tf.train.get_global_step())
    #
    # errors = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # for i in range(1):
        (probs_res, loss_res, p_estim, r_estim) = \
            sess.run((probs, loss, p, r), feed_dict={sample_data: x})
        # errors.append(loss_res)
        # print(i)
    
    print(np.nanmean(np.abs(r_estim - df.r) / np.fmax(r_estim, df.r)))
