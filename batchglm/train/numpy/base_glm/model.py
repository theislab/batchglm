import abc
import logging
from typing import Union

import dask.array
import numpy as np

from .vars import ModelVarsGlm

logger = logging.getLogger("batchglm")


class ModelIwls:
    """
    Class for maintaining state of IWLS updates.

    Attributes
    ----------
    model_vars : ModelVarsGlm
        Model variables.
    """

    def __init__(self, model_vars: ModelVarsGlm):
        self.model_vars = model_vars

    @property
    def converged(self):
        return self.model_vars.converged

    @converged.setter
    def converged(self, value):
        self.model_vars.converged = value

    @property
    def idx_not_converged(self):
        return self.model_vars.idx_not_converged

    @property
    def theta_location(self):
        return self.model_vars.theta_location

    @theta_location.setter
    def theta_location(self, value):
        self.model_vars.theta_location = value

    @property
    def theta_scale(self):
        return self.model_vars.theta_scale

    @theta_scale.setter
    def theta_scale(self, value):
        self.model_vars.theta_scale = value

    def theta_scale_j_setter(self, value, j):
        self.model_vars.theta_scale_j_setter(value=value, j=j)

    @abc.abstractmethod
    def fim_weight(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def ll(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def ll_j(self, j) -> np.ndarray:
        pass

    @property
    def ll_byfeature(self) -> Union[np.ndarray, dask.array.core.Array]:
        return np.sum(self.ll, axis=0)

    def ll_byfeature_j(self, j) -> Union[np.ndarray, dask.array.core.Array]:
        return np.sum(self.ll_j(j=j), axis=0)

    @abc.abstractmethod
    def fim_weight_location_location(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def ybar(self) -> Union[np.ndarray, dask.array.core.Array]:
        pass

    @abc.abstractmethod
    def fim_weight_location_location_j(self, j) -> np.ndarray:
        pass

    @abc.abstractmethod
    def jac_weight(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def jac_weight_j(self, j) -> np.ndarray:
        pass

    @abc.abstractmethod
    def ybar_j(self, j) -> Union[np.ndarray, dask.array.core.Array]:
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
        xh = np.matmul(self.design_loc, self.constraints_loc)
        return np.einsum("fob,oc->fbc", np.einsum("ob,of->fob", xh, w), xh)

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
    def hessian_weight_location_location(self) -> np.ndarray:
        pass

    @property
    def hessian_location_location(self) -> np.ndarray:
        """

        :return: (features x inferred param x inferred param)
        """
        w = self.hessian_weight_location_location
        xh = np.matmul(self.design_loc, self.constraints_loc)
        return np.einsum("fob,oc->fbc", np.einsum("ob,of->fob", xh, w), xh)

    @abc.abstractmethod
    def hessian_weight_location_scale(self) -> np.ndarray:
        pass

    @property
    def hessian_location_scale(self) -> np.ndarray:
        """

        :return: (features x inferred param x inferred param)
        """
        w = self.hessian_weight_location_scale
        return np.einsum(
            "fob,oc->fbc",
            np.einsum("ob,of->fob", np.matmul(self.design_loc, self.constraints_loc), w),
            np.matmul(self.design_scale, self.constraints_scale),
        )

    @abc.abstractmethod
    def hessian_weight_scale_scale(self) -> np.ndarray:
        pass

    @property
    def hessian_scale_scale(self) -> np.ndarray:
        """

        :return: (features x inferred param x inferred param)
        """
        w = self.hessian_weight_scale_scale
        xh = np.matmul(self.design_scale, self.constraints_scale)
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
        xh = np.matmul(self.design_loc, self.constraints_loc)  # (observations x inferred param)
        return np.einsum("fob,of->fb", np.einsum("ob,of->fob", xh, w), ybar)

    def jac_location_j(self, j) -> np.ndarray:
        """

        :return: (features x inferred param)
        """
        # Make sure that dimensionality of sliced array is kept:
        if isinstance(j, int) or isinstance(j, np.int32) or isinstance(j, np.int64):
            j = [j]
        w = self.fim_weight_location_location_j(j=j)  # (observations x features)
        ybar = self.ybar_j(j=j)  # (observations x features)
        xh = np.matmul(self.design_loc, self.constraints_loc)  # (observations x inferred param)
        return np.einsum("fob,of->fb", np.einsum("ob,of->fob", xh, w), ybar)

    @property
    def jac_scale(self) -> Union[np.ndarray, dask.array.core.Array]:
        """

        :return: (features x inferred param)
        """
        w = self.jac_weight_scale  # (observations x features)
        xh = np.matmul(self.design_scale, self.constraints_scale)  # (observations x inferred param)
        return np.einsum("fob,of->fb", np.einsum("ob,of->fob", xh, w), xh)

    def jac_scale_j(self, j) -> np.ndarray:
        """

        :return: (features x inferred param)
        """
        # Make sure that dimensionality of sliced array is kept:
        if isinstance(j, int) or isinstance(j, np.int32) or isinstance(j, np.int64):
            j = [j]
            w = self.jac_weight_scale_j(j=j)  # (observations x features)
            xh = np.matmul(self.design_scale, self.constraints_scale)  # (observations x inferred param)
            return np.einsum("fob,of->fb", np.einsum("ob,of->fob", xh, w), xh)
