import abc
import numpy as np
import logging

logger = logging.getLogger("batchglm")


class ModelIwls:

    def __init__(
            self,
            model_vars
    ):
        self.model_vars = model_vars
        #self.params = np.concatenate(
        #    [
        #        model_vars.init_a_clipped,
        #        model_vars.init_b_clipped,
        #    ],
        #    axis=0
        #)

    @property
    def a_var(self):
        return self.model_vars.a_var

    @a_var.setter
    def a_var(self, value):
        self.model_vars.a_var = value

    @property
    def b_var(self):
        return self.model_vars.b_var

    @a_var.setter
    def b_var(self, value):
        self.model_vars.b_var = value

    @abc.abstractmethod
    def fim_weight(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def ll(self) -> np.ndarray:
        pass

    @property
    def ll_byfeature(self) -> np.ndarray:
        return np.sum(self.ll, axis=0)

    @abc.abstractmethod
    def fim_weight(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def ybar(self) -> np.ndarray:
        pass

    @property
    def fim(self) -> np.ndarray:
        """

        :return: (features x inferred param x inferred param)
        """
        w = self.fim_weight  # (observations x features)
        # constraints: (observed param x inferred param)
        # design: (observations x observed param)
        # w: (observations x features)
        # fim: (features x inferred param x inferred param)
        return np.einsum(
            'fob,oc->fbc',
            np.einsum('bo,of->fob', np.matmul(self.constraints_loc.T, self.design_loc.T), w),
            np.matmul(self.design_loc, self.constraints_loc)
        )

    def jac(self) -> np.ndarray:
        """

        :return: (features x inferred param)
        """
        w = self.fim_weight  # (observations x features)
        ybar = self.model.ybar  # (observations x features)
        xh = np.matmul(self.design_loc, self.constraints_loc)  # (observations x inferred param)
        return np.einsum(
            'fob,of->fb',
            np.einsum('ob,of->fob', xh, w),
            ybar
        )
