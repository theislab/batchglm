import abc
from typing import Union

import dask.array
import numpy as np
import scipy.sparse


class ModelVarsGlm:
    """
    Build variables to be optimzed and their constraints.

    Attributes
    ----------
    constraints_scale : Union[np.ndarray, dask.array.core.Array]
        Scale model constraints for VGLM fitting
    constraints_loc : Union[np.ndarray, dask.array.core.Array]
        Location model constraints for VGLM fitting
    theta_location : np.ndarray
        Location model parameters
    theta_scale : np.ndarray
        Scale model parameters
    converged : np.ndarray
        Whether or not given parameters are converged
    params : Union[np.ndarray, dask.array.core.Array]
        Model parameters
    converged : np.ndarray
        Whether or not a parameter has converged
    npar_a : int
    dtype : str
    n_features : int
    idx_train_loc : np.ndarray
        Training indices for location model
    idx_train_scale : np.ndarray
        Training indices for scale model
    """

    constraints_loc: Union[np.ndarray, dask.array.core.Array]
    constraints_scale: Union[np.ndarray, dask.array.core.Array]
    params: Union[np.ndarray, dask.array.core.Array]
    converged: np.ndarray
    npar_a: int
    dtype: str
    n_features: int

    def __init__(
        self,
        init_location: Union[np.ndarray, dask.array.core.Array],
        init_scale: Union[np.ndarray, dask.array.core.Array],
        constraints_loc: Union[np.ndarray, dask.array.core.Array],
        constraints_scale: Union[np.ndarray, dask.array.core.Array],
        chunk_size_genes: int,
        dtype: str,
    ):
        """
        :param init_location:
            Initialisation for all parameters of mean model. (mean model size x features)
        :param init_scale:
            Initialisation for all parameters of dispersion model. (dispersion model size x features)
        :param constraints_scale:
            Scale model constraints for VGLM fitting
        :param constraints_loc:
            Location model constraints for VGLM fitting
        :param chunk_size_genes:
            chunk size for dask
        :param dtype:
            Precision used in tensorflow.
        """
        self.constraints_loc = np.asarray(constraints_loc, dtype)
        self.constraints_scale = np.asarray(constraints_scale, dtype)

        init_location_clipped = self.np_clip_param(np.asarray(init_location, dtype=dtype), "theta_location")
        init_scale_clipped = self.np_clip_param(np.asarray(init_scale, dtype=dtype), "theta_scale")
        self.params = dask.array.from_array(
            np.concatenate(
                [
                    init_location_clipped,
                    init_scale_clipped,
                ],
                axis=0,
            ),
            chunks=(1000, chunk_size_genes),
        )
        self.npar_a = init_location_clipped.shape[0]

        # Properties to follow gene-wise convergence.
        self.converged = np.repeat(a=False, repeats=self.params.shape[1])  # Initialise to non-converged.

        self.dtype = dtype
        self.n_features = self.params.shape[1]
        self.idx_train_loc = np.arange(0, init_location.shape[0])
        self.idx_train_scale = np.arange(init_location.shape[0], init_location.shape[0] + init_scale.shape[0])

    @property
    def idx_not_converged(self):
        """Find which features are not converged"""
        return np.where(np.logical_not(self.converged))[0]

    @property
    def theta_location(self):
        """Location parameters"""
        theta_location = self.params[0 : self.npar_a]
        return self.np_clip_param(theta_location, "theta_location")

    @theta_location.setter
    def theta_location(self, value):
        # Threshold new entry:
        value = self.np_clip_param(value, "theta_location")
        # Write either new dask array or into numpy array:
        if isinstance(self.params, dask.array.core.Array):
            temp = self.params.compute()
            temp[0 : self.npar_a] = value
            self.params = dask.array.from_array(temp, chunks=self.params.chunksize)
        else:
            self.params[0 : self.npar_a] = value

    @property
    def theta_scale(self):
        """Scale parameters"""
        theta_scale = self.params[self.npar_a :]
        return self.np_clip_param(theta_scale, "theta_scale")

    @theta_scale.setter
    def theta_scale(self, value):
        # Threshold new entry:
        value = self.np_clip_param(value, "theta_scale")
        # Write either new dask array or into numpy array:
        if isinstance(self.params, dask.array.core.Array):
            temp = self.params.compute()
            temp[self.npar_a :] = value
            self.params = dask.array.from_array(temp, chunks=self.params.chunksize)
        else:
            self.params[self.npar_a :] = value

    def theta_scale_j_setter(self, value, j):
        """Setter ofr a specific theta_scale value."""
        # Threshold new entry:
        value = self.np_clip_param(value, "theta_scale")
        # Write either new dask array or into numpy array:
        if isinstance(self.params, dask.array.core.Array):
            temp = self.params.compute()
            temp[self.npar_a :, j] = value
            self.params = dask.array.from_array(temp, chunks=self.params.chunksize)
        else:
            self.params[self.npar_a :, j] = value

    @abc.abstractmethod
    def param_bounds(self, dtype):
        pass
