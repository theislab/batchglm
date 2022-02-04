import abc

try:
    import anndata
except ImportError:
    anndata = None
import numpy as np

from .external import _ModelGLM


class Model(_ModelGLM, metaclass=abc.ABCMeta):

    """Generalized Linear Model (GLM) with normal noise."""

    def link_loc(self, data):
        """Short summary.

        :param type data: Description of parameter `data`.
        :return: Description of returned object.
        :rtype: type

        """
        return data

    def inverse_link_loc(self, data):
        """Short summary.

        :param type data: Description of parameter `data`.
        :return: Description of returned object.
        :rtype: type

        """
        return data

    def link_scale(self, data):
        """Short summary.

        :param type data: Description of parameter `data`.
        :return: Description of returned object.
        :rtype: type

        """
        return np.log(data)

    def inverse_link_scale(self, data):
        """Short summary.

        :param type data: Description of parameter `data`.
        :return: Description of returned object.
        :rtype: type

        """
        return np.exp(data)

    @property
    def eta_loc(self) -> np.ndarray:
        """Short summary.

        :return: Description of returned object.
        :rtype: np.ndarray

        """
        eta = np.matmul(self.design_loc, self.a)
        if self.size_factors is not None:
            eta *= np.expand_dims(self.size_factors, axis=1)
        return eta

    def eta_loc_j(self, j) -> np.ndarray:
        """Short summary.

        :param type j: Description of parameter `j`.
        :return: Description of returned object.
        :rtype: np.ndarray

        """
        # Make sure that dimensionality of sliced array is kept:
        if isinstance(j, int) or isinstance(j, np.int32) or isinstance(j, np.int64):
            j = [j]
        eta = np.matmul(self.design_loc, self.a[:, j])
        if self.size_factors is not None:
            eta *= np.expand_dims(self.size_factors, axis=1)
        eta = self.np_clip_param(eta, "eta_loc")
        return eta

    # Re-parameterizations:

    @property
    def mean(self) -> np.ndarray:
        """Short summary.

        :return: Description of returned object.
        :rtype: np.ndarray

        """
        return self.location

    @property
    def sd(self) -> np.ndarray:
        """Short summary.

        :return: Description of returned object.
        :rtype: np.ndarray

        """
        return self.scale
