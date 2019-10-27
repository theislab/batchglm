import abc
import numpy as np
import scipy

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
            max_steps: int
    ):
        # Iterate until conditions are fulfilled.
        train_step = 0

        # Set all to convergence status = False, this is needed if multiple
        # training strategies are run:
        converged_current = np.repeat(False, repeats=self.model.model_vars.n_features)

        ll_current = -np.inf
        print("iter %i: ll=%f" % (0, np.sum(self.model.ll_byfeature)))
        while np.any(np.logical_not(converged_current)) and train_step < max_steps:
            ll_previous = ll_current
            self.a_var = self.a_var + self.iwls_step()
            ll_current = self.model.ll_byfeature
            #features_updated = self.model.model_vars.updated
            ll_converged = (ll_previous - ll_current) / ll_previous < pkg_constants.LLTOL_BY_FEATURE
            converged_f = np.logical_and(np.logical_not(converged_current), ll_converged)
            train_step += 1
            print("iter %i: ll=%f" % (train_step, np.sum(ll_current)))
            self.lls.append(ll_current)

    def iwls_step(self) -> np.ndarray:
        """

        :return: (features x inferred param)
        """
        w = self.model.fim_weight  # (observations x features)
        ybar = self.model.ybar  # (observations x features)
        # Translate to problem of form ax = b for each feature:
        # (in the following, X=design and Y=counts)
        # a=X^T*W*X: ([features] x inferred param)
        # x=theta: ([features] x inferred param)
        # b=X^T*W*Ybar: ([features] x inferred param)
        xh = np.matmul(self.design_loc, self.constraints_loc)  # (observations x inferred param)
        xhw = np.einsum('ob,of->fob', xh, w)  # (features x observations x inferred param)
        a = np.einsum('fob,ob->fbb', xhw, xh),
        b = np.einsum('fob,of->fb', xhw, ybar),
        delta_theta = np.concatenate([
            np.expand_dims(np.linalg.lstsq(a[i], b[i]), axis=0)
            for i in range(a.shape[0])
        ], axis=0)
        return delta_theta

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
        # Create standard executable model:
        self.model = self.get_model_container(self.input_data)
        self.model._a_var = self.a_var
        self.model._b_var = self.b_var

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

