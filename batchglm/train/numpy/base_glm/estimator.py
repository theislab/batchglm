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
        while np.any(converged_current) and train_step < max_steps:
            ll_previous = ll_current
            self.a_var = self.a_var + self.iwls_step()
            ll_current = self.model.ll
            #features_updated = self.model.model_vars.updated
            ll_converged = (ll_previous - ll_current) / ll_previous < pkg_constants.LLTOL_BY_FEATURE
            converged_f = np.logical_and(np.logical_not(converged_current), ll_converged)
            train_step += 1
            self.lls.append(ll_current)

    def iwls_step(self):
        w = np.diag(self.model.fim_weight)
        xw = np.matmul(self.x.T, w)
        delta_theta = np.linalg.lstsq(
            np.matmul(xw, self.x),
            xw
        )
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

