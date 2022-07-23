import numpy as np
from scipy.linalg.lapack import dgeqrf, dormqr, dtrtrs

from .external import _ModelGLM


class QRDecomposition:
    def __init__(self, nrows: int, ncoefs: int, cur_x):

        self.nr = nrows
        self.nc = ncoefs
        self.x = cur_x
        # self.x_copy = np.zeros((nrows * ncoefs), dtype=float)
        # self.tau = np.zeros(ncoefs)
        # self.effects = np.zeros(nrows)
        self.weights = 1.0  # np.zeros(nrows)
        self.lwork_geqrf = -1
        self.lwork_ormqr = -1
        self.trans_trtrs = 0
        self.trans_ormqr = "T"
        self.side = "L"

        self.x_copy, self.tau, tmpwork, info = dgeqrf(a=np.zeros((self.nr, self.nc)), lwork=self.lwork_geqrf)
        self.lwork_geqrf = tmpwork + 0.5
        if self.lwork_geqrf < 1:
            self.lwork_geqrf = 1
        # work_geqrf = # TODO work_geqrf.resize(lwork_geqrf);

        # Repeating for dormqr
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
        # work_ormqr = # TODO work_ormqr.resize(lwork_ormqr);

    def store_weights(self, w=None):
        if w is None:
            self.weights = 1.0
        else:
            self.weights = np.sqrt(w)

    def decompose(self):
        self.x_copy = self.x.copy()
        self.x_copy *= self.weights

        self.x_copy, self.tau, tmpwork, info = dgeqrf(a=self.x_copy, lwork=self.lwork_geqrf)
        print(self.x_copy.shape, self.tau.shape)
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

        values = np.broadcast_to(
            np.log(sum_norm_x / sum_weights), (n_obs, n_features)
        ).compute()  # shape(n_obs, n_features)

        for j in range(n_features):
            qr.solve(values[:, j])
            output[:, j] = qr.effects[:n_parm, 0]

    else:
        """
         {
        const bool weights_are_the_same=allw.is_row_repeated();
        if (weights_are_the_same && num_tags) {
            QR.store_weights(allw.get_row(0));
            QR.decompose();
        }

        // Finding the delta.
        double delta=0;
        if (counts.is_data_integer()) {
            Rcpp::IntegerMatrix imat=counts.get_raw_int();
            delta=*std::max_element(imat.begin(), imat.end());
        } else {
            Rcpp::NumericMatrix dmat=counts.get_raw_dbl();
            delta=*std::max_element(dmat.begin(), dmat.end());
        }
        delta=std::min(delta, 1.0/6);

        for (int tag=0; tag<num_tags; ++tag) {
            if (!weights_are_the_same) {
                QR.store_weights(allw.get_row(tag));
                QR.decompose();
            }
            counts.fill_row(tag, current.data());
            const double* optr=allo.get_row(tag);

            // Computing normalized log-expression values.
            for (int lib=0; lib<num_libs; ++lib) {
                current[lib]=std::log(std::max(delta, current[lib])) - optr[lib];
            }

            // Performing the QR decomposition and taking the solution.
            QR.solve(current.data());
            auto curout=output.row(tag);
            std::copy(QR.effects.begin(), QR.effects.begin()+num_coefs, curout.begin());
        }
        """
        raise NotImplementedError("This method is not yet implemented.")
    return output
