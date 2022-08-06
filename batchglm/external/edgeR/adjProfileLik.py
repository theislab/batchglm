import numpy as np
import scipy

from .estimator import NBEstimator


def adjusted_profile_likelihood(
    estimator: NBEstimator,
    adjust: bool = True,
):
    """
    Featurewise Cox-Reid adjusted profile log-likelihoods for the dispersion.
    dispersion can be a scalar or a featurewise vector.
    Computationally, dispersion can also be a matrix, but the apl is still computed tagwise.
    y is a matrix: rows are genes/tags/transcripts, columns are samples/libraries.
    offset is a matrix of the same dimensions as y.
    This is a numpy vectorized python version of edgeR's adjProfileLik function implemented in C++.
    """
    low_value = 1e-10
    log_low_value = np.log(low_value)

    estimator.train(maxit=250, tolerance=1e-10)
    model = estimator._model_container
    poisson_idx = np.where(1 / model.scale < 0)[0].compute()

    if len(poisson_idx) == model.num_features:
        loglik = model.x * np.log(model.location) - model.location - scipy.special.lgamma(model.x + 1)
    elif len(poisson_idx) == 0:
        loglik = model.ll
    else:
        loglik = np.zeros_like(model.x)

        poisson_x = model.x[:, poisson_idx]
        poisson_loc = model.location_j(poisson_idx)

        loglik[:, poisson_idx] = poisson_x * np.log(poisson_loc) - poisson_loc - scipy.special.lgamma(poisson_x + 1)

        non_poisson_idx = np.where(model.theta_scale > 0)[0]
        loglik[:, non_poisson_idx] = model.ll_j(non_poisson_idx)

    sum_loglik = np.sum(loglik, axis=0)

    if adjust:
        w = -model.fim_weight_location_location

        adj = np.zeros(model.num_features)
        n_loc_params = model.design_loc.shape[1]
        if n_loc_params == 1:
            adj = np.sum(w, axis=0)
            adj = np.log(np.abs(adj)).compute()
        else:
            xh = model.xh_loc
            xhw = np.einsum("ob,of->fob", xh, w)
            fim = np.einsum("fob,oc->fbc", xhw, xh).compute()
            for i in range(fim.shape[0]):

                ldu, _, info = scipy.linalg.lapack.dsytrf(lower=0, a=fim[i])
                if info < 0:
                    adj[i] = 0
                    print(f"LDL factorization failed for feature {i}")
                else:
                    ldu_diag = np.diag(ldu)
                    adj[i] = np.sum(
                        np.where((ldu_diag < low_value) | np.isinf(ldu_diag), log_low_value, np.log(ldu_diag))
                    )

        adj /= 2
        sum_loglik -= adj

    return sum_loglik
