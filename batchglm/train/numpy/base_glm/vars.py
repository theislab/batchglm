import numpy as np
import tensorflow as tf
import abc


class ModelVarsGlm:
    """ Build variables to be optimzed and their constraints.

    a_var and b_var slices of the tf1.Variable params which contains
    all parameters to be optimized during model estimation.
    Params is defined across both location and scale model so that
    the hessian can be computed for the entire model.
    a and b are the clipped parameter values which also contain
    constraints and constrained dependent coefficients which are not
    directly optimized.
    """

    constraints_loc: np.ndarray
    constraints_scale: np.ndarray
    params: np.ndarray
    a_var: np.ndarray
    b_var: np.ndarray
    converged: np.ndarray
    npar_a: int
    dtype: str
    n_features: int

    def __init__(
            self,
            init_a: np.ndarray,
            init_b: np.ndarray,
            constraints_loc: np.ndarray,
            constraints_scale: np.ndarray,
            dtype: str
    ):
        """

        :param init_a: nd.array (mean model size x features)
            Initialisation for all parameters of mean model.
        :param init_b: nd.array (dispersion model size x features)
            Initialisation for all parameters of dispersion model.
        :param dtype: Precision used in tensorflow.
        """
        self.constraints_loc = np.asarray(constraints_loc, dtype)
        self.constraints_scale = np.asarray(constraints_scale, dtype)

        init_a_clipped = self.np_clip_param(np.asarray(init_a, dtype=dtype), "a_var")
        init_b_clipped = self.np_clip_param(np.asarray(init_b, dtype=dtype), "b_var")
        self.params = np.concatenate(
            [
                init_a_clipped,
                init_b_clipped,
            ],
            axis=0
        )
        self.npar_a = init_a_clipped.shape[0]

        # Properties to follow gene-wise convergence.
        self.converged = np.repeat(a=False, repeats=self.params.shape[1])  # Initialise to non-converged.

        self.dtype = dtype
        self.n_features = self.params.shape[1]
        self.idx_train_loc = np.arange(0, init_a.shape[0])
        self.idx_train_scale = np.arange(init_a.shape[0], init_a.shape[0] + init_b.shape[0])

    @property
    def idx_not_converged(self):
        return np.where(np.logical_not(self.converged))[0]

    @property
    def a_var(self):
        a_var = self.params[0:self.npar_a]
        return self.np_clip_param(a_var, "a_var")

    @a_var.setter
    def a_var(self, value):
        self.params[0:self.npar_a] = value

    @property
    def b_var(self):
        b_var = self.params[self.npar_a:]
        return self.np_clip_param(b_var, "b_var")

    @b_var.setter
    def b_var(self, value):
        self.params[self.npar_a:] = value

    def b_var_j_setter(self, value, j):
        self.params[self.npar_a:, j] = value

    @abc.abstractmethod
    def param_bounds(self, dtype):
        pass
