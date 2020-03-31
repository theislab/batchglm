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
    def converged(self):
        return self.model_vars.converged

    @property
    def idx_not_converged(self):
        return self.model_vars.idx_not_converged

    @converged.setter
    def converged(self, value):
        self.model_vars.converged = value

    @property
    def a_var(self):
        return self.model_vars.a_var

    @a_var.setter
    def a_var(self, value):
        self.model_vars.a_var = value

    @property
    def b_var(self):
        return self.model_vars.b_var

    @b_var.setter
    def b_var(self, value):
        self.model_vars.b_var = value

    def b_var_j_setter(self, value, j):
        self.model_vars.b_var_j_setter(value=value, j=j)

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
    def ll_byfeature(self) -> np.ndarray:
        return np.sum(self.ll, axis=0)

    def ll_byfeature_j(self, j) -> np.ndarray:
        return np.sum(self.ll_j(j=j), axis=0)

    @abc.abstractmethod
    def fim_weight_aa(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def ybar(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def fim_weight_aa_j(self, j) -> np.ndarray:
        pass

    @abc.abstractmethod
    def jac_weight(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def jac_weight_j(self, j) -> np.ndarray:
        pass

    @abc.abstractmethod
    def ybar_j(self, j) -> np.ndarray:
        pass

    @property
    def fim_aa(self) -> np.ndarray:
        """
        Location-location coefficient block of FIM

        :return: (features x inferred param x inferred param)
        """
        w = self.fim_weight_aa  # (observations x features)
        # constraints: (observed param x inferred param)
        # design: (observations x observed param)
        # w: (observations x features)
        # fim: (features x inferred param x inferred param)
        xh = np.matmul(self.design_loc, self.constraints_loc)
        return np.einsum(
            'fob,oc->fbc',
            np.einsum('ob,of->fob', xh, w),
            xh
        )

    @abc.abstractmethod
    def fim_ab(self) -> np.ndarray:
        pass

    @property
    def fim_bb(self) -> np.ndarray:
        pass

    @property
    def fim(self) -> np.ndarray:
        """
        Full FIM

        :return: (features x inferred param x inferred param)
        """
        fim_aa = self.fim_aa
        fim_bb = self.fim_bb
        fim_ab = self.fim_ab
        fim_ba = np.transpose(fim_ab, axes=[0, 2, 1])
        return - np.concatenate([
            np.concatenate([fim_aa, fim_ab], axis=2),
            np.concatenate([fim_ba, fim_bb], axis=2)
        ], axis=1)

    @abc.abstractmethod
    def hessian_weight_aa(self) -> np.ndarray:
        pass

    @property
    def hessian_aa(self) -> np.ndarray:
        """

        :return: (features x inferred param x inferred param)
        """
        w = self.hessian_weight_aa
        xh = np.matmul(self.design_loc, self.constraints_loc)
        return np.einsum(
            'fob,oc->fbc',
            np.einsum('ob,of->fob', xh, w),
            xh
        )

    @abc.abstractmethod
    def hessian_weight_ab(self) -> np.ndarray:
        pass

    @property
    def hessian_ab(self) -> np.ndarray:
        """

        :return: (features x inferred param x inferred param)
        """
        w = self.hessian_weight_ab
        return np.einsum(
            'fob,oc->fbc',
            np.einsum('ob,of->fob', np.matmul(self.design_loc, self.constraints_loc), w),
            np.matmul(self.design_scale, self.constraints_scale)
        )

    @abc.abstractmethod
    def hessian_weight_bb(self) -> np.ndarray:
        pass

    @property
    def hessian_bb(self) -> np.ndarray:
        """

        :return: (features x inferred param x inferred param)
        """
        w = self.hessian_weight_bb
        xh = np.matmul(self.design_scale, self.constraints_scale)
        return np.einsum(
            'fob,oc->fbc',
            np.einsum('ob,of->fob', xh, w),
            xh
        )

    @property
    def hessian(self) -> np.ndarray:
        """

        :return: (features x inferred param x inferred param)
        """
        h_aa = self.hessian_aa
        h_bb = self.hessian_bb
        h_ab = self.hessian_ab
        h_ba = np.transpose(h_ab, axes=[0, 2, 1])
        return np.concatenate([
            np.concatenate([h_aa, h_ab], axis=2),
            np.concatenate([h_ba, h_bb], axis=2)
        ], axis=1)

    @property
    def jac(self) -> np.ndarray:
        return np.concatenate([self.jac_a, self.jac_b], axis=-1)

    @property
    def jac_a(self) -> np.ndarray:
        """

        :return: (features x inferred param)
        """
        w = self.fim_weight_aa  # (observations x features)
        ybar = self.ybar  # (observations x features)
        xh = np.matmul(self.design_loc, self.constraints_loc)  # (observations x inferred param)
        return np.einsum(
            'fob,of->fb',
            np.einsum('ob,of->fob', xh, w),
            ybar
        )

    def jac_a_j(self, j) -> np.ndarray:
        """

        :return: (features x inferred param)
        """
        # Make sure that dimensionality of sliced array is kept:
        if isinstance(j, int) or isinstance(j, np.int32) or isinstance(j, np.int64):
            j = [j]
        w = self.fim_weight_aa_j(j=j)  # (observations x features)
        ybar = self.ybar_j(j=j)  # (observations x features)
        xh = np.matmul(self.design_loc, self.constraints_loc)  # (observations x inferred param)
        return np.einsum(
            'fob,of->fb',
            np.einsum('ob,of->fob', xh, w),
            ybar
        )

    @property
    def jac_b(self) -> np.ndarray:
        """

        :return: (features x inferred param)
        """
        w = self.jac_weight_b  # (observations x features)
        xh = np.matmul(self.design_scale, self.constraints_scale)  # (observations x inferred param)
        return np.einsum(
            'fob,of->fb',
            np.einsum('ob,of->fob', xh, w),
            xh
        )

    def jac_b_j(self, j) -> np.ndarray:
        """

        :return: (features x inferred param)
        """
        # Make sure that dimensionality of sliced array is kept:
        if isinstance(j, int) or isinstance(j, np.int32) or isinstance(j, np.int64):
            j = [j]
            w = self.jac_weight_b_j(j=j)  # (observations x features)
            xh = np.matmul(self.design_scale, self.constraints_scale)  # (observations x inferred param)
            return np.einsum(
                'fob,of->fb',
                np.einsum('ob,of->fob', xh, w),
                xh
            )
