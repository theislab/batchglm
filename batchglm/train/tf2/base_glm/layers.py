from typing import Union

import abc
import tensorflow as tf
tf.keras.backend.set_floatx("float64")

from .processModel import ProcessModelGLM


class UnpackParamsGLM(tf.keras.layers.Layer, ProcessModelGLM):

    """
    Layer that slices the parameter tensor into mean and variance block.
    """

    def __init__(self):
        super(UnpackParamsGLM, self).__init__()

    def call(self, inputs, **kwargs):
        """
        :param inputs: tuple (params, border)
           Must contain the parameter matrix (params) and the first index
           of the variance block within the parameters matrix (border)

        :return tf.Tensor, tf.Tensor
           The two returned tensor correspond to the mean and variance block
           of the parameter matrix.
        """
        params, border = inputs
        a_var = params[0:border]  # loc obs
        b_var = params[border:]  # scale obs
        a_var = self.tf_clip_param(a_var, "a_var")
        b_var = self.tf_clip_param(b_var, "b_var")
        return a_var, b_var


class LinearLocGLM(tf.keras.layers.Layer, ProcessModelGLM):

    """
    Computes the dot product between the design matrix of the mean model and the mean block of the parameter matrix.
    """

    def __init__(self):
        super(LinearLocGLM, self).__init__()

    def _eta_loc(
            self,
            a_var: tf.Tensor,
            design_loc: tf.Tensor,
            constraints_loc: Union[tf.Tensor, None] = None,
            size_factors: Union[tf.Tensor, None] = None
    ):
        """
        Does the actual computation of eta_loc.

        :param a_var: tf.Tensor
            the mean block of the parameter matrix
        :param design_loc: tf.Tensor
            the design matrix of the mean model
        :param contraints_loc: tf.Tensor, optional
            ??? # TODO
        :param size_factors: tf.Tensor, optional
            ??? # TODO

        :return tf.Tensor
            the mean values for each individual distribution, encoded in linker space.
        """
        if constraints_loc is not None:
            eta_loc = tf.matmul(
                design_loc,
                tf.matmul(constraints_loc, a_var)
            )
        else:
            eta_loc = tf.matmul(design_loc, a_var)

        if size_factors is not None and size_factors.shape != (1, 1):
            eta_loc = self.with_size_factors(eta_loc, size_factors)

        eta_loc = self.tf_clip_param(eta_loc, "eta_loc")

        return eta_loc

    @abc.abstractmethod
    def with_size_factors(self, eta_loc, size_factors):
        """
        Calculates eta_loc with size_factors. Is noise model specific and needs to be implemented in the inheriting
        layer.
        :param eta_loc: tf.Tensor
            the mean values for each individual distribution, encoded in linker space
        """

    def call(self, inputs, **kwargs):
        """
        Calculates the eta_loc tensor, containing the mean values for each individual distribution,
        encoded in linker space.

        :param input: tuple
            Must contain a_var, design_loc, constraints_loc and size_factors in this order, where
            contraints_loc and size_factor can be None.

        :return tf.Tensor
            the mean values for each individual distribution, encoded in linker space.
        """
        return self._eta_loc(*inputs)


class LinearScaleGLM(tf.keras.layers.Layer, ProcessModelGLM):

    """
    Computes the dot product between the design matrix of the variance model
    and the variance block of the parameter matrix.
    """

    def __init__(self):
        super(LinearScaleGLM, self).__init__()

    def _eta_scale(
            self,
            b_var: tf.Tensor,
            design_scale: tf.Tensor,
            constraints_scale: Union[tf.Tensor, None] = None
    ):
        """
        Does the actual computation of eta_scale.

        :param b_var: tf.Tensor
            the variance block of the parameter matrix
        :param design_scale: tf.Tensor
            the design matrix of the mean model
        :param contraints_scale: tf.Tensor, optional
            ??? # TODO

        :return tf.Tensor
            the variance values for each individual distribution, encoded in linker space.
        """
        if constraints_scale is not None:
            eta_scale = tf.matmul(
                design_scale,
                tf.matmul(constraints_scale, b_var)
            )
        else:
            eta_scale = tf.matmul(design_scale, b_var)

        eta_scale = self.tf_clip_param(eta_scale, "eta_scale")

        return eta_scale

    def call(self, inputs, **kwargs):
        """
        Calculates the eta_scale tensor, containing the variance values for each individual distribution,
        encoded in linker space.

        :param input: tuple
            Must contain b_var, design_scale and constraints_loc in this order, where
            contraints_loc can be None.

        :return tf.Tensor
            the variance values for each individual distribution, encoded in linker space.
        """
        return self._eta_scale(*inputs)


class LinkerLocGLM(tf.keras.layers.Layer):

    """
    Translation from linker to data space for the mean model.
    """

    def __init__(self):
        super(LinkerLocGLM, self).__init__()

    @abc.abstractmethod
    def _inv_linker(self, loc: tf.Tensor):
        """
        Translates the given mean values from linker to data space. Depends on the given noise model and needs to
        be implemented in the inheriting layer.

        :param loc: tf. Tensor
            the mean values for each individual distribution, encoded in linker space.

        :return tf.Tensor
            the mean values for each individual distribution, encoded in data space.
        """

    def call(self, eta_loc: tf.Tensor, **kwargs):
        """
        Calls the distribution specific linker function to translate from linker to data space.

        :param eta_loc: tf.Tensor
            the mean values for each individual distribution, encoded in linker space.

        :return tf.Tensor
            the mean values for each individual distribution, encoded in data space.
        """
        loc = self._inv_linker(eta_loc)
        return loc


class LinkerScaleGLM(tf.keras.layers.Layer):

    """
    Translation from linker to data space for the variance model.
    """

    def __init__(self):
        super(LinkerScaleGLM, self).__init__()

    @abc.abstractmethod
    def _inv_linker(self, scale: tf.Tensor):
        pass

    def call(self, eta_scale: tf.Tensor, **kwargs):
        """
        Calls the distribution specific linker function to translate from linker to data space.

        :param eta_scale: tf.Tensor
            the variance values for each individual distribution, encoded in linker space.

        :return tf.Tensor
            the variance values for each individual distribution, encoded in data space.
        """
        scale = self._inv_linker(eta_scale)
        return scale


class LikelihoodGLM(tf.keras.layers.Layer, ProcessModelGLM):

    """
    Contains the computation of the distribution specific log-likelihood function
    """

    def __init__(self, dtype):
        super(LikelihoodGLM, self).__init__()
        self.ll_dtype = dtype

    @abc.abstractmethod
    def _ll(self, eta_loc, eta_scale, loc, scale, x, n_features):
        """
        Does the actual likelihood calculation. Depends on the given noise model and needs to be implemented in the
        inheriting layer.

        :param eta_loc: tf.Tensor
            the mean values for each individual distribution, encoded in linker space.
        :param eta_scale: tf.Tensor
            the variance values for each individual distribution, encoded in linker space.
        :param loc: tf.Tensor
            the mean values for each individual distribution, encoded in data space.
        :param scale: tf.Tensor
            the variance values for each individual distribution, encoded in data space.
        :param x: tf.Tensor
            the input data
        :param n_features
            number of features.

        :return tf.Tensor
            the log-likelihoods of each individual data point.
        """

    def call(self, inputs, **kwargs):
        """
        Calls the distribution specific log-likelihood function.

        :param inputs: tuple
            Must contain eta_loc, eta_scale, loc, scale, x, n_features in this order.

        :return tf.Tensor
            the log-likelihoods of each individual data point.
        """
        return self._ll(*inputs)
