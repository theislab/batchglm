import abc
from typing import Any, Callable, Union

import dask.array
import numpy as np

from ...base import BaseModelContainer
from ....models.base_glm import ModelGLM


def dask_compute(func: Callable):
    def func_wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result.compute() if isinstance(result, dask.array.core.Array) else result

    return func_wrapper


class NumpyModelContainer(BaseModelContainer):
    """
    Build variables to be optimized.

    Attributes
    ----------
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
    idx_train_loc : np.ndarray
        Training indices for location model
    idx_train_scale : np.ndarray
        Training indices for scale model
    npar_location : int
        number of location parameters
    dtype : str
        data type to be used
    """

    params: Union[np.ndarray, dask.array.core.Array]
    converged: np.ndarray
    npar_location: int
    dtype: str

    def __init__(
        self,
        model,
        init_theta_location: Union[np.ndarray, dask.array.core.Array],
        init_theta_scale: Union[np.ndarray, dask.array.core.Array],
        chunk_size_genes: int,
        dtype: str,
    ):
        """
        :param init_location:
            Initialisation for all parameters of mean model. (mean model size x features)
        :param init_scale:
            Initialisation for all parameters of dispersion model. (dispersion model size x features)
        :param chunk_size_genes:
            chunk size for dask
        :param dtype:
            Precision used in tensorflow.
        """

        self._model = model
        init_theta_location_clipped = model.np_clip_param(
            np.asarray(init_theta_location, dtype=dtype), "theta_location"
        )
        init_theta_scale_clipped = model.np_clip_param(np.asarray(init_theta_scale, dtype=dtype), "theta_scale")
        self.params = dask.array.from_array(
            np.concatenate(
                [
                    init_theta_location_clipped,
                    init_theta_scale_clipped,
                ],
                axis=0,
            ),
            chunks=(1000, chunk_size_genes),
        )
        self.npar_location = init_theta_location_clipped.shape[0]

        # Properties to follow gene-wise convergence.
        self.converged = np.repeat(a=False, repeats=self.params.shape[1])  # Initialise to non-converged.

        self.dtype = dtype
        self.idx_train_loc = np.arange(0, init_theta_location.shape[0])
        self.idx_train_scale = np.arange(
            init_theta_location.shape[0], init_theta_location.shape[0] + init_theta_scale.shape[0]
        )

        # overriding the location and scale parameter by referencing the getter functions within the properties.
        self._model._theta_location_getter = self._theta_location_getter
        self._model._theta_scale_getter = self._theta_scale_getter

    # Is this actually used in diffxpy? Why?
    @property
    def niter(self):
        return None

    # Is this actually used in diffxpy? Why?
    @property
    def error_codes(self):
        return np.array()

    @property
    def model(self) -> ModelGLM:
        return self._model

    @property
    def fisher_inv(self) -> np.ndarray:
        return self._fisher_inv

    def _theta_location_getter(self) -> dask.array.core.Array:
        theta_location = self.params[0 : self.npar_location]
        return self.np_clip_param(theta_location, "theta_location")

    def _theta_scale_getter(self) -> dask.array.core.Array:
        theta_scale = self.params[self.npar_location :]
        return self.np_clip_param(theta_scale, "theta_scale")

    def __getattr__(self, attr: str):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError()
        return self.model.__getattribute__(attr)

    @property
    def idx_not_converged(self) -> np.ndarray:
        """Find which features are not converged"""
        return np.where(np.logical_not(self.converged))[0]

    @property
    def theta_location(self) -> dask.array.core.Array:
        """Location parameters"""
        return self._theta_location_getter()

    @theta_location.setter
    def theta_location(self, value):
        # Threshold new entry:
        value = self.np_clip_param(value, "theta_location")
        # Write either new dask array or into numpy array:
        if isinstance(self.params, dask.array.core.Array):
            temp = self.params.compute()
            temp[0 : self.npar_location] = value
            self.params = dask.array.from_array(temp, chunks=self.params.chunksize)
        else:
            self.params[0 : self.npar_location] = value

    @property
    def theta_scale(self) -> dask.array.core.Array:
        """Scale parameters"""
        return self._theta_scale_getter()

    @theta_scale.setter
    def theta_scale(self, value):
        # Threshold new entry:
        value = self.np_clip_param(value, "theta_scale")
        # Write either new dask array or into numpy array:
        if isinstance(self.params, dask.array.core.Array):
            temp = self.params.compute()
            temp[self.npar_location :] = value
            self.params = dask.array.from_array(temp, chunks=self.params.chunksize)
        else:
            self.params[self.npar_location :] = value

    @property
    def theta_location_constrained(self) -> Union[np.ndarray, dask.array.core.Array]:
        """dot product of location constraints with location parameter giving new constrained parameters"""
        return np.dot(self.constraints_loc, self.theta_location)

    @property
    def theta_scale_constrained(self) -> Union[np.ndarray, dask.array.core.Array]:
        """dot product of scale constraints with scale parameter giving new constrained parameters"""
        return np.dot(self.constraints_scale, self.theta_scale)

    def theta_scale_j(self, j) -> dask.array.core.Array:
        if isinstance(j, int) or isinstance(j, np.int32) or isinstance(j, np.int64):
            j = [j]
        return self.np_clip_param(self.params[self.npar_location :, j], "theta_scale")

    def theta_scale_j_setter(self, value, j):
        """Setter ofr a specific theta_scale value."""
        # Threshold new entry:
        value = self.np_clip_param(value, "theta_scale")
        # Write either new dask array or into numpy array:
        if isinstance(self.params, dask.array.core.Array):
            temp = self.params.compute()
            temp[self.npar_location :, j] = value
            self.params = dask.array.from_array(temp, chunks=self.params.chunksize)
        else:
            self.params[self.npar_location :, j] = value

    # jacobians

    @abc.abstractmethod
    def jac_weight(self) -> Union[np.ndarray, dask.array.core.Array]:
        pass

    @abc.abstractmethod
    def jac_weight_j(self, j) -> Union[np.ndarray, dask.array.core.Array]:
        pass

    @property
    def jac(self) -> Union[np.ndarray, dask.array.core.Array]:
        return np.concatenate([self.jac_location, self.jac_scale], axis=-1)

    @property
    def jac_location(self) -> Union[np.ndarray, dask.array.core.Array]:
        """

        :return: (features x inferred param)
        """
        w = self.fim_weight_location_location  # (observations x features)
        ybar = self.ybar  # (observations x features)
        xh = self.xh_loc  # (observations x inferred param)
        inner = np.einsum("ob,of->fob", xh, w)
        return np.einsum("fob,of->fb", inner, ybar)

    def jac_location_j(self, j) -> Union[np.ndarray, dask.array.core.Array]:
        """

        :return: (features x inferred param)
        """
        # Make sure that dimensionality of sliced array is kept:
        if isinstance(j, int) or isinstance(j, np.int32) or isinstance(j, np.int64):
            j = [j]
        w = self.fim_weight_location_location_j(j=j)  # (observations x features)
        ybar = self.ybar_j(j=j)  # (observations x features)
        xh = self.xh_loc  # (observations x inferred param)
        return np.einsum("fob,of->fb", np.einsum("ob,of->fob", xh, w), ybar)

    @property
    def jac_scale(self) -> Union[np.ndarray, dask.array.core.Array]:
        """

        :return: (features x inferred param)
        """
        w = self.jac_weight_scale  # (observations x features)
        xh = self.xh_scale  # (observations x inferred param)
        return np.einsum("fob,of->fb", np.einsum("ob,of->fob", xh, w), xh)

    @abc.abstractmethod
    def jac_weight_scale_j(self, j) -> Union[np.ndarray, dask.array.core.Array]:
        pass

    @dask_compute
    def jac_scale_j(self, j) -> np.ndarray:
        """

        :return: (features x inferred param)
        """
        # Make sure that dimensionality of sliced array is kept:
        if isinstance(j, int) or isinstance(j, np.int32) or isinstance(j, np.int64):
            j = [j]
        w = self.jac_weight_scale_j(j=j)  # (observations x features)
        xh = self.xh_scale  # (observations x inferred param)
        return np.einsum("fob,of->fb", np.einsum("ob,of->fob", xh, w), xh)

    # hessians

    @property
    @abc.abstractmethod
    def hessian_weight_location_location(self) -> Union[np.ndarray, dask.array.core.Array]:
        pass

    @property
    def hessian_location_location(self) -> Union[np.ndarray, dask.array.core.Array]:
        """

        :return: (features x inferred param x inferred param)
        """
        w = self.hessian_weight_location_location
        xh = self.xh_loc
        return np.einsum("fob,oc->fbc", np.einsum("ob,of->fob", xh, w), xh)

    @property
    @abc.abstractmethod
    def hessian_weight_location_scale(self) -> Union[np.ndarray, dask.array.core.Array]:
        pass

    @property
    def hessian_location_scale(self) -> Union[np.ndarray, dask.array.core.Array]:
        """

        :return: (features x inferred param x inferred param)
        """
        w = self.hessian_weight_location_scale
        return np.einsum("fob,oc->fbc", np.einsum("ob,of->fob", self.xh_loc, w), self.xh_scale)

    @property
    @abc.abstractmethod
    def hessian_weight_scale_scale(self) -> Union[np.ndarray, dask.array.core.Array]:
        pass

    @property
    def hessian_scale_scale(self) -> Union[np.ndarray, dask.array.core.Array]:
        """

        :return: (features x inferred param x inferred param)
        """
        w = self.hessian_weight_scale_scale
        xh = self.xh_scale
        return np.einsum("fob,oc->fbc", np.einsum("ob,of->fob", xh, w), xh)

    @property
    def hessian(self) -> Union[np.ndarray, dask.array.core.Array]:
        """

        :return: (features x inferred param x inferred param)
        """
        h_aa = self.hessian_location_location
        h_bb = self.hessian_scale_scale
        h_ab = self.hessian_location_scale
        h_ba = np.transpose(h_ab, axes=[0, 2, 1])
        return np.concatenate([np.concatenate([h_aa, h_ab], axis=2), np.concatenate([h_ba, h_bb], axis=2)], axis=1)

    # fim

    @abc.abstractmethod
    def fim_weight_location_location_j(self, j) -> Union[np.ndarray, dask.array.core.Array]:
        pass

    @property
    def fim_location_location(self) -> Union[np.ndarray, dask.array.core.Array]:
        """
        Location-location coefficient block of FIM

        :return: (features x inferred param x inferred param)
        """
        w = self.fim_weight_location_location  # (observations x features)
        # constraints: (observed param x inferred param)
        # design: (observations x observed param)
        # w: (observations x features)
        # fim: (features x inferred param x inferred param)
        xh = self.xh_loc
        return np.einsum("fob,oc->fbc", np.einsum("ob,of->fob", xh, w), xh)

    @property
    @abc.abstractmethod
    def fim_location_scale(self) -> np.ndarray:
        pass

    @property
    def fim_scale_scale(self) -> np.ndarray:
        pass

    @property
    def fim(self) -> Union[np.ndarray, dask.array.core.Array]:
        """
        Full FIM

        :return: (features x inferred param x inferred param)
        """
        fim_location_location = self.fim_location_location
        fim_scale_scale = self.fim_scale_scale
        fim_location_scale = self.fim_location_scale
        fim_ba = np.transpose(fim_location_scale, axes=[0, 2, 1])
        return -np.concatenate(
            [
                np.concatenate([fim_location_location, fim_location_scale], axis=2),
                np.concatenate([fim_ba, fim_scale_scale], axis=2),
            ],
            axis=1,
        )

    @abc.abstractmethod
    def fim_weight(self) -> Union[np.ndarray, dask.array.core.Array]:
        pass

    @property
    @abc.abstractmethod
    def fim_weight_location_location(self) -> Union[np.ndarray, dask.array.core.Array]:
        pass

    @property
    @abc.abstractmethod
    def ll(self) -> Union[np.ndarray, dask.array.core.Array]:
        pass

    @abc.abstractmethod
    def ll_j(self, j) -> Union[np.ndarray, dask.array.core.Array]:
        pass

    @property  # type: ignore
    @dask_compute
    def ll_byfeature(self) -> np.ndarray:
        return np.sum(self.ll, axis=0)

    @dask_compute
    def ll_byfeature_j(self, j) -> np.ndarray:
        return np.sum(self.ll_j(j=j), axis=0)

    # bar

    @property
    @abc.abstractmethod
    def ybar(self) -> Union[np.ndarray, dask.array.core.Array]:
        pass

    @abc.abstractmethod
    def ybar_j(self, j) -> Union[np.ndarray, dask.array.core.Array]:
        pass
