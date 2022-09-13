import dask.array
import numpy as np
from scipy.linalg.lapack import dgeqrf, dormqr, dtrtrs

from .external import _ModelGLM


class QRDecomposition:
    def __init__(self, nrows: int, ncoefs: int, cur_x):
        self.nr = nrows
        self.nc = ncoefs
        self.x = cur_x
        self.weights = 1.0  # np.zeros(nrows)
        self.lwork_geqrf = -1
        self.lwork_ormqr = -1
        self.trans_trtrs = 0
        self.trans_ormqr = "T"
        self.side = "L"

        # workspace queries calling LAPACK subroutines via scipy.linalg.lapack
        self.x_copy, self.tau, tmpwork, info = dgeqrf(a=np.zeros((self.nr, self.nc)), lwork=self.lwork_geqrf)
        self.lwork_geqrf = tmpwork + 0.5
        if self.lwork_geqrf < 1:
            self.lwork_geqrf = 1

        self.effects, tmpwork, info = dormqr(
            side=self.side,
            trans=self.trans_ormqr,
            a=self.x_copy,
            tau=self.tau,
            c=np.zeros((ncoefs, 1)),
            lwork=self.lwork_ormqr,
        )
        self.lwork_ormqr = tmpwork + 0.5
        if self.lwork_ormqr < 1:
            self.lwork_ormqr = 1

    def store_weights(self, w=None):
        if w is None:
            self.weights = 1.0
        else:
            self.weights = np.sqrt(w)

    def decompose(self):
        self.x_copy = self.x.copy()
        self.x_copy *= self.weights

        self.x_copy, self.tau, tmpwork, info = dgeqrf(a=self.x_copy, lwork=self.lwork_geqrf)
        if info != 0:
            raise RuntimeError("QR decomposition failed")

    def solve(self, y):
        self.effects = (y * self.weights)[..., None]
        self.effects, tmpwork, info = dormqr(
            side="L", trans="T", a=self.x_copy, tau=self.tau, c=self.effects, lwork=self.lwork_ormqr
        )
        if info != 0:
            raise RuntimeError("Q**T multiplication failed")
        self.effects, info = dtrtrs(lower=0, trans=0, unitdiag=0, a=self.x_copy, b=self.effects)
        if info != 0:
            raise RuntimeError("failed to solve the triangular system")


def get_levenberg_start(model: _ModelGLM, disp: np.ndarray, use_null: bool):
    """
    Parameter initialisation of location parameters using QR decomposition.
    This method is a python version of the C++ code in edgeR.
    :param model: A GLM model object
    :param disp: the fixed dispersion parameter used during the calculation.
    :param use_null: ??? this must be true, the other is not implemented
    """

    n_obs = model.num_observations
    n_features = model.num_features
    n_parm = model.num_loc_params
    model_weights = np.ones(n_features, dtype=float)  # TODO ### integrate into model

    qr = QRDecomposition(n_obs, n_parm, model.design_loc)
    output = np.zeros((n_parm, model.num_features), dtype=float)  # shape (n_parm, n_features)

    if use_null:

        qr.store_weights(w=None)
        qr.decompose()

        if model.size_factors is None:
            sf_exp = np.ones((n_obs, 1), dtype=float)
        else:
            sf_exp = np.exp(model.size_factors)  # shape (n_obs, 1)
        weights = model_weights * sf_exp / (1.0 + disp * sf_exp)  # shape (n_obs, n_features)
        sum_norm_x = np.sum(model.x * weights / sf_exp, axis=0)  # shape (n_features,)
        sum_weights = np.sum(weights, axis=0)  # shape (n_features,)

        values = np.broadcast_to(np.log(sum_norm_x / sum_weights), (n_obs, n_features))
        if isinstance(values, dask.array.core.Array):
            values = values.compute()  # shape(n_obs, n_features)

        for j in range(n_features):
            qr.solve(values[:, j])
            output[:, j] = qr.effects[:n_parm, 0]

    else:
        raise NotImplementedError("This method is not yet implemented.")
    return output
