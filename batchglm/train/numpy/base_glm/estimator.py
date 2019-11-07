import abc
import logging
import numpy as np
import pprint
import scipy
import scipy.optimize

from .external import _EstimatorGLM, pkg_constants
from .training_strategies import TrainingStrategies

logger = logging.getLogger("batchglm")


class EstimatorGlm(_EstimatorGLM, metaclass=abc.ABCMeta):
    """
    Estimator for Generalized Linear Models (GLMs).
    """

    def __init__(
            self,
            model,
            input_data,
            dtype,
    ):
        if input_data.design_scale.shape[1] != 1:
            raise ValueError("cannot model more than one scale parameter with numpy backend right now.")
        _EstimatorGLM.__init__(
            self=self,
            model=model,
            input_data=input_data
        )
        self.dtype = dtype
        self.values = []
        self.lls = []

        self.TrainingStrategies = TrainingStrategies

    def initialize(self):
        pass

    def train(
            self,
            max_steps: int,
            update_b_freq: int = 5
    ):
        # Iterate until conditions are fulfilled.
        train_step = 0
        delayed_converged = np.tile(False, self.model.model_vars.n_features)

        ll_current = - self.model.ll_byfeature
        logging.getLogger("batchglm").debug("iter %i: ll=%f" % (0, np.sum(ll_current)))
        while np.any(np.logical_not(delayed_converged)) and \
                train_step < max_steps:
            # Update parameters:
            # Line search step for scale model:
            if train_step % update_b_freq == 0 and train_step > 0:
                b_var_cache = self.model.b_var.copy()
                self.model.b_var = self.b_step(idx=np.where(np.logical_not(delayed_converged))[0])
                # Reverse update by feature if update leads to worse loss:
                ll_proposal = - self.model.ll_byfeature
                b_var_new = self.model.b_var.copy()
                b_var_new[:, ll_proposal > ll_current] = b_var_cache[:, ll_proposal > ll_current]
                self.model.b_var = b_var_new
                delayed_b_converged = self.model.converged.copy()
            # IWLS step for location model:
            self.model.a_var = self.model.a_var + self.iwls_step()

            # Evaluate convergence
            ll_previous = ll_current
            ll_current = - self.model.ll_byfeature
            converged_f = (ll_previous - ll_current) / ll_previous < pkg_constants.LLTOL_BY_FEATURE
            # Location model convergence status has to be updated if b model was updated
            if train_step % update_b_freq == 0 and train_step > 0:
                self.model.converged = converged_f
                delayed_converged = converged_f
            else:
                self.model.converged = np.logical_or(self.model.converged, converged_f)
            train_step += 1
            logging.getLogger("batchglm").debug(
                "iter %i: ll=%f, converged: %i" %
                (train_step, np.sum(ll_current), np.sum(self.model.converged))
            )
            self.lls.append(ll_current)

    def iwls_step(self) -> np.ndarray:
        """

        :return: (inferred param x features)
        """
        w = self.model.fim_weight_j(j=self.model.idx_not_converged)  # (observations x features)
        ybar = self.model.ybar_j(j=self.model.idx_not_converged)  # (observations x features)
        # Translate to problem of form ax = b for each feature:
        # (in the following, X=design and Y=counts)
        # a=X^T*W*X: ([features] x inferred param)
        # x=theta: ([features] x inferred param)
        # b=X^T*W*Ybar: ([features] x inferred param)
        xh = np.matmul(self.model.design_loc, self.model.constraints_loc)
        xhw = np.einsum('ob,of->fob', xh, w)
        a = np.einsum('fob,oc->fbc', xhw, xh)
        b = np.einsum('fob,of->fb', xhw, ybar)
        # Via np.linalg.solve:
        delta_theta = np.zeros_like(self.model.a_var)
        delta_theta[:, self.model.idx_not_converged] = np.linalg.solve(a, b).T
        # Via np.linalg.lsts:
        #delta_theta[:, self.idx_not_converged] = np.concatenate([
        #    np.expand_dims(np.linalg.lstsq(a[i, :, :], b[i, :])[0], axis=-1)
        #    for i in self.idx_not_converged)
        #], axis=-1)
        # Via np.linalg.inv:
        # #delta_theta[:, self.idx_not_converged] = np.concatenate([
        #    np.expand_dims(np.matmul(np.linalg.inv(a[i, :, :]), b[i, :]), axis=-1)
        #    for i in self.idx_not_converged)
        #], axis=-1)
        return delta_theta

    def b_step(
            self,
            idx: np.ndarray,
            linesearch: bool = False
    ) -> np.ndarray:
        """

        :return:
        """
        x0 = -10

        def cost_b_var(x):
            self.model.b_var_j_setter(value=x, j=j)
            return - np.sum(self.model.ll_j(j=j))

        def grad_b_var(x):
            self.model.b_var_j_setter(value=x, j=j)
            return - self.model.jac_b_j(j=j)

        b_var_new = self.model.b_var.copy()
        for j in idx:
            if linesearch:
                ls_result = scipy.optimize.line_search(
                    f=cost_b_var,
                    myfprime=grad_b_var,
                    xk=np.array([x0]),
                    pk=np.array([1.]),
                    gfk=None,
                    old_fval=None,
                    old_old_fval=None,
                    args=(),
                    c1=0.0001,
                    c2=0.9,
                    amax=50.,
                    extra_condition=None,
                    maxiter=1000
                )
                b_var_new[0, j] = x0 + ls_result[0]
            else:
                ls_result = scipy.optimize.minimize_scalar(
                    fun=cost_b_var,
                    args=(),
                    method='brent',
                    tol=None,
                    options={'maxiter': 500}
                )
                b_var_new[0, j] = ls_result["x"]

        return b_var_new

    def finalize(self):
        """
        Evaluate all tensors that need to be exported from session and save these as class attributes
        and close session.

        Changes .model entry from tf1-based EstimatorGraph to numpy based Model instance and
        transfers relevant attributes.
        """
        # Read from numpy-IRLS estimator specific model:

        self._hessian = self.model.hessian
        self._fisher_inv = np.linalg.inv(- self._hessian)
        self._jacobian = np.sum(np.abs(self.model.jac / self.model.x.shape[0]), axis=1)
        self._log_likelihood = self.model.ll_byfeature
        self._loss = np.sum(self._log_likelihood)

    @abc.abstractmethod
    def get_model_container(
            self,
            input_data
    ):
        pass

    @abc.abstractmethod
    def init_par(
            self,
            input_data,
            init_a,
            init_b,
            init_model
    ):
        pass

