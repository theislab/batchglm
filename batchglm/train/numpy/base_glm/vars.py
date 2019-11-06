import dask.array
import numpy as np
import scipy.sparse
import abc


class ModelVarsGlm:
    """
    Build variables to be optimzed and their constraints.

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
            chunk_size_genes: int,
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
        self.params = dask.array.from_array(np.concatenate(
            [
                init_a_clipped,
                init_b_clipped,
            ],
            axis=0
        ), chunks=(1000, chunk_size_genes))
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
        # Threshold new entry:
        value = self.np_clip_param(value, "a_var")
        # Write either new dask array or into numpy array:
        if isinstance(self.params, dask.array.core.Array):
            temp = self.params.compute()
            temp[0:self.npar_a] = value
            self.params = dask.array.from_array(temp, chunks=self.params.chunksize)
        else:
            self.params[0:self.npar_a] = value

    @property
    def b_var(self):
        b_var = self.params[self.npar_a:]
        return self.np_clip_param(b_var, "b_var")

    @b_var.setter
    def b_var(self, value):
        # Threshold new entry:
        value = self.np_clip_param(value, "b_var")
        # Write either new dask array or into numpy array:
        if isinstance(self.params, dask.array.core.Array):
            temp = self.params.compute()
            temp[self.npar_a:] = value
            self.params = dask.array.from_array(temp, chunks=self.params.chunksize)
        else:
            self.params[self.npar_a:] = value

    def b_var_j_setter(self, value, j):
        # Threshold new entry:
        value = self.np_clip_param(value, "b_var")
        # Write either new dask array or into numpy array:
        if isinstance(self.params, dask.array.core.Array):
            temp = self.params.compute()
            temp[self.npar_a:, j] = value
            self.params = dask.array.from_array(temp, chunks=self.params.chunksize)
        else:
            self.params[self.npar_a:, j] = value

    @abc.abstractmethod
    def param_bounds(self, dtype):
        pass
