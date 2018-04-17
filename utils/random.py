import tensorflow as tf
import numpy as np

from tensorflow.contrib.distributions import NegativeBinomial

from tensorflow.contrib import eager as tfe


def negative_binomial(r, mu, as_numpy_array=True):
    # ugly hack using tensorflow, since parametrisation with `p`
    # does not work with np.random.negative_binomial

    retVal = None
    with tf.Graph().as_default():
        r = tf.convert_to_tensor(r, dtype=tf.float64)
        mu = tf.convert_to_tensor(mu, dtype=tf.float64)
        r = tf.cast(r, tf.float64)
        mu = tf.cast(mu, tf.float64)

        p = mu / (r + mu)

        dist = NegativeBinomial(total_count=r, probs=p)

        retVal = tf.squeeze(dist.sample(1))

        # run sampling session
        if not tfe.in_eager_mode():
            with tf.Session() as sess:
                retVal = sess.run(retVal)

    # random_data = np.random.negative_binomial(
    #     self.r,
    #     self.p,
    # )
    # return random_data

    if as_numpy_array:
        retVal = np.asarray(retVal)

    return retVal
