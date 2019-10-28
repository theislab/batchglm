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
    updated: np.ndarray
    converged: np.ndarray
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

        self.init_a = np.asarray(init_a, dtype=dtype)
        self.init_b = np.asarray(init_b, dtype=dtype)

        self.init_a_clipped = self.np_clip_param(self.init_a, "a_var")
        self.init_b_clipped = self.np_clip_param(self.init_b, "b_var")

        # Param is the only tf1.Variable in the graph.
        # a_var and b_var have to be slices of params.
        self.params = np.concatenate(
            [
                self.init_a_clipped,
                self.init_b_clipped,
            ],
            axis=0
        )

        # Properties to follow gene-wise convergence.
        self.updated = np.repeat(a=True, repeats=self.params.shape[1])  # Initialise to is updated.
        self.converged = np.repeat(a=False, repeats=self.params.shape[1])  # Initialise to non-converged.

        self.dtype = dtype
        self.n_features = self.params.shape[1]
        self.idx_train_loc = np.arange(0, init_a.shape[0])
        self.idx_train_scale = np.arange(init_a.shape[0], init_a.shape[0] + init_b.shape[0])

    @property
    def a_var(self):
        a_var = self.params[0:self.init_a.shape[0]]
        return self.np_clip_param(a_var, "a_var")

    @a_var.setter
    def a_var(self, value):
        self.params[0:self.init_a.shape[0]] = value

    @property
    def b_var(self):
        b_var = self.params[self.init_a.shape[0]:]
        return self.np_clip_param(b_var, "b_var")

    @b_var.setter
    def b_var(self, value):
        self.params[self.init_a.shape[0]:] = value

    @abc.abstractmethod
    def param_bounds(self, dtype):
        pass

    def convergence_update(self, status):
        self.converged = status
