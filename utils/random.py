import numpy as np
import scipy.stats as scistats


# import tensorflow as tf
# import impl.tf.negative_binomial.util as nb_utils
# from tensorflow.contrib import eager as tfe


class NegativeBinomial:
    mean: np.ndarray
    # variance: np.ndarray
    # p: np.ndarray
    r: np.ndarray

    def __init__(self, r=None, variance=None, p=None, mean=None):
        if r is not None:
            if variance is not None:
                raise ValueError("Must pass either shape 'r' or variance, but not both")

            if p is not None:
                if mean is not None:
                    raise ValueError("Must pass either probs or means, but not both")

                mean = p * r / (1 - p)
                # variance = mean / (1 - p)

            elif mean is not None:
                if p is not None:
                    raise ValueError("Must pass either probs or means, but not both")

                # p = mean / (r + mean)
                # variance = mean + (np.square(mean) / r)

            else:
                raise ValueError("Must pass probs or means")

        elif variance is not None:
            if r is not None:
                raise ValueError("Must pass either shape 'r' or variance, but not both")

            if p is not None:
                if mean is not None:
                    raise ValueError("Must pass either probs or means, but not both")

                mean = variance * (1 - p)
                r = mean / (variance - mean)

            elif mean is not None:
                if p is not None:
                    raise ValueError("Must pass either probs or means, but not both")

                # p = 1 - (mean / variance)
                r = mean * mean / (variance - mean)
            else:
                raise ValueError("Must pass probs or means")
        else:
            raise ValueError("Must pass shape 'r' or variance")

        self.mean = mean
        # self.variance = variance
        # self.p = p
        self.r = r

    @property
    def variance(self):
        return self.mean + (np.square(self.mean) / self.r)

    @property
    def p(self):
        return self.mean / (self.r + self.mean)

    def sample(self, size=None):
        # # ugly hack using tensorflow, since parametrisation with `p`
        # # does not work with np.random.negative_binomial
        #
        # with tf.Graph().as_default():
        #     mean = tf.convert_to_tensor(self.mean, dtype=tf.float64)
        #     variance = tf.convert_to_tensor(self.variance, dtype=tf.float64)
        #     mu = tf.cast(mean, tf.float64)
        #     variance = tf.cast(variance, tf.float64)
        #     import tensorflow.contrib.distributions as distributions
        #     distributions.NegativeBinomial()
        #     dist = nb_utils.NegativeBinomial(mean=mean, variance=variance)
        #
        #     retVal = tf.squeeze(dist.sample(size))
        #
        #     # run sampling session
        #     if not tfe.in_eager_mode():
        #         with tf.Session() as sess:
        #             retVal = sess.run(retVal)
        # if as_numpy_array:
        #     retVal = np.asarray(retVal)
        #
        # return retVal
        # ####

        # numpy uses an alternative parametrization
        # see also https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
        random_data = np.random.negative_binomial(
            n=self.r,
            p=1 - self.p,
            size=size
        )
        return random_data

    def prob(self, sample_data):
        return scistats.nbinom(n=self.r, p=1 - self.p).pmf(sample_data)

    def log_prob(self, sample_data):
        return scistats.nbinom(n=self.r, p=1 - self.p).logpmf(sample_data)
