import tensorflow as tf
import tensorflow.contrib as tfcontrib

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

################################
# Estimate NB distribution parameters by parameter optimization
################################

if __name__ == '__main__':
    # load sample data
    x = np.loadtxt("sample_data.tsv", delimiter="\t")
    df = pd.read_csv("sample_params.tsv", sep="\t")

    smpls = np.array(range(10))
    x = x[:, smpls]
    df = df.iloc[smpls]

    # previously sampled data
    sample_data = tf.placeholder(tf.float32)

    # sample_data.shape: (N,M)
    N = tf.shape(sample_data)[0]
    N = tf.to_float(N)

    # distribution parameters which should be optimized
    r = tf.Variable(np.repeat(10.0, x.shape[1]), dtype=tf.float32, name="r")

    # keep mu constant
    mu = tf.reduce_mean(sample_data, axis=0)
    p = mu / (r + mu)

    distribution = tfcontrib.distributions.NegativeBinomial(total_count=r,
                                                            probs=p,
                                                            name="nb-dist")
    probs = distribution.log_prob(sample_data)

    # minimize negative log probability (log(1) = 0)
    loss = -tf.reduce_sum(probs, name="loss")

    train_op = tf.train.AdamOptimizer(learning_rate=0.05)
    train_op = train_op.minimize(loss, global_step=tf.train.get_global_step())

    errors = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(10000):
            (probs_res, loss_res, p_estim, r_estim, _) = \
                sess.run((probs, loss, p, r, train_op), feed_dict={sample_data: x})
            errors.append(loss_res)
            print(i)

    print(np.nanmean(np.abs(r_estim - df.r) / np.fmax(r_estim, df.r)))
