import abc
import numpy as np
import scipy
import scipy.optimize

from .external import _EstimatorGLM, pkg_constants


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
        _EstimatorGLM.__init__(
            self=self,
            model=model,
            input_data=input_data
        )
        self.dtype = dtype
        self.values = []
        self.lls = []

    def initialize(self):
        pass

    def train(
            self,
            max_steps: int,
            update_b_freq: int = 5,
            tr_a: float = 0.1
    ):
        # Iterate until conditions are fulfilled.
        train_step = 0

        # Set all to convergence status = False, this is needed if multiple
        # training strategies are run:
        converged_current = np.repeat(False, repeats=self.model.model_vars.n_features)

        ll_current = - self.model.ll_byfeature
        print("iter %i: ll=%f" % (0, np.sum(ll_current)))
        while np.any(np.logical_not(converged_current)) and train_step < max_steps:
            ll_previous = ll_current
            # Line search step for scale model:
            if train_step % update_b_freq == 0 and train_step > 0:
                self.model.b_var = self.b_step()
                ll_current = - self.model.ll_byfeature
                print("iter %i: ll=%f, converged: %i" % (train_step, np.sum(ll_current), np.sum(converged_current)))
            # IWLS step for location model:
            delta_a = np.zeros_like(self.model.a_var)
            delta_a[:, np.logical_not(converged_current)] += self.iwls_step()[:, np.logical_not(converged_current)]
            self.model.a_var = self.model.a_var + delta_a
            ll_current = - self.model.ll_byfeature
            #features_updated = self.model.model_vars.updated
            ll_converged = (ll_previous - ll_current) / ll_previous < pkg_constants.LLTOL_BY_FEATURE
            converged_f = np.logical_and(np.logical_not(converged_current), ll_converged)
            train_step += 1
            converged_current = np.logical_or(converged_current, ll_converged)
            print("iter %i: ll=%f, converged: %i" % (train_step, np.sum(ll_current), np.sum(converged_current)))
            self.lls.append(ll_current)
        # Line search step for scale model:
        self.model.b_var = self.b_step()
        ll_current = - self.model.ll_byfeature
        print("iter %i: ll=%f, converged: %i" % (train_step, np.sum(ll_current), np.sum(converged_current)))

    def iwls_step(self) -> np.ndarray:
        """

        :return: (inferred param x features)
        """
        w = self.model.fim_weight  # (observations x features)
        ybar = self.model.ybar  # (observations x features)
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
        delta_theta = np.linalg.solve(a, b).T
        # Via np.linalg.lsts:
        #delta_theta = np.concatenate([
        #    np.expand_dims(np.linalg.lstsq(a[i, :, :], b[i, :])[0], axis=-1)
        #    for i in range(a.shape[0])
        #], axis=-1)
        # Via np.linalg.inv:
        # #delta_theta = np.concatenate([
        #    np.expand_dims(np.matmul(np.linalg.inv(a[i, :, :]), b[i, :]), axis=-1)
        #    for i in range(a.shape[0])
        #], axis=-1)
        return delta_theta

    def b_step(
            self,
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

        b_var_new = np.zeros_like(self.model.b_var)
        for j in range(self.model.b_var.shape[1]):
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
        self._hessian = - self.model.fim
        self._fisher_inv = np.linalg.inv(- self._hessian)
        self._jacobian = self.model.jac
        self._log_likelihood = self.model.ll
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

