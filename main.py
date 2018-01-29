import tensorflow as tf
import tensorflow.contrib as tfcontrib

import pandas as pd
import numpy as np
from tensorflow.contrib.distributions.python.ops import negative_binomial

import RSAgraph


# import matplotlib.pyplot as plt

################################
# Estimate NB distribution parameters by parameter optimization
################################

if __name__ == '__main__':
    # load sample data
    x = np.loadtxt("resources/sample_data.tsv", delimiter="\t")
    df = pd.read_csv("resources/sample_params.tsv", sep="\t")

    smpls = np.array(range(10))
    x = x[:, smpls]
    df = df.iloc[smpls]

    # previously sampled data


    RSAgraph.buildGraph()
    
    errors = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer(), feed_dict={sample_data: x})
        # initialize_all_variables(feed_dict={sample_data: x})

        for i in range(10000):
            (probs_res, r_estim, p_estim, loss_res, _) = \
                sess.run((probs, distribution.total_count, distribution.probs, loss, train_op),
                         feed_dict={sample_data: x})
            errors.append(loss_res)
            print(i)
    tf.reset_default_graph()

    print(np.nanmean(np.abs(r_estim - df.r) / np.fmax(r_estim, df.r)))
