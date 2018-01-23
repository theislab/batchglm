#!/usr/bin/env python3

import tensorflow as tf
import tensorflow.contrib as tfcontrib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

################################
# Generate sample data
################################

p_real = tf.random_uniform([10000], 0, 1)
r_real = tf.random_uniform([10000], 10, 100)
r_real = tf.round(r_real)
# mu = r_real / (r_real + p_real)
# logit_prob = tf.log(mu / (1 - mu))

random_data = tfcontrib.distributions.NegativeBinomial(total_count=r_real, probs=p_real)

# run sampling session
x = None
with tf.Session() as sess:
    (p, r, x) = sess.run([p_real, r_real, random_data.sample(500)])

print(x)

# create example plot + print values
plt.hist(x[:, 9977], bins=50)
plt.show()
print(p[1])
print(r[1])

# save sample data
np.savetxt("sample_data.tsv", x, delimiter="\t")
df = pd.DataFrame({"p": p, "r": r})
df.to_csv("sample_params.tsv", sep="\t", index=False)

