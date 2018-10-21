import numpy as np
# import scipy.stats
# import scipy.special
from scipy.special import gammaln

try:
    import xarray as xr
except ImportError:
    xr = None


class NegativeBinomial:
    r"""
    Negative binomial distribution.
    This class supports re-parameterising, sampling and calculation of
    probabilities of negative binomial distributed data.
    """

    mean: np.ndarray
    # variance: np.ndarray
    # p: np.ndarray
    r: np.ndarray

    @classmethod
    def mme(cls, data, axis=0):
        r"""
        Fit a Negative Binomial distribution to `data`.
        Uses the closed-form Method-of-Moments to estimate the dispersion.

        :param data: The data
        :param axis: Axis along which the distributions are provided
        :return: `NegativeBinomial` object

        """
        mean = np.mean(data, axis=axis)
        variance = np.mean(np.square(data - mean), axis=axis)

        return cls(mean=mean, variance=variance)

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
        """
        Sample from all distributions data of size `size`.
        :param size: The size
        :return: numpy array containing sampled data

        """
        # numpy uses an alternative parametrization
        # see also https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
        random_data = np.random.negative_binomial(
            n=self.r,
            p=1 - self.p,
            size=size
        )
        return random_data

    def prob(self, X):
        """
        Calculate the probability of each value in `X` given this distribution

        :param X: The data
        :return: numpy array of probabilitites

        """
        # p = self.p
        # r = self.r
        # return scipy.stats.nbinom(n=r, p=1 - p).pmf(X)
        # return binom(X + r - 1, X) * np.power(p, X) * np.power(1 - p, r)
        return np.exp(self.log_prob(X))

    def log_prob(self, X):
        """
        Calculate the log-probability of each value in `X` given this distribution

        :param X: The data
        :return: numpy array of log-probabilitites

        """
        p = self.p
        r = self.r

        # broadcasting
        if xr != None and isinstance(p, xr.DataArray) and isinstance(r, xr.DataArray) and isinstance(X, xr.DataArray):
            p, r, X = xr.align(p, r, X)
        else:
            p, r, X = np.broadcast_arrays(p, r, X)

        # return scipy.stats.nbinom(n=r, p=1 - p).logpmf(X)
        coeff = gammaln(r + X) - gammaln(X + 1) - gammaln(r)
        return coeff + r * np.log(1 - p) + X * np.log(p)
