import sys
import time

import dask.array
import numpy as np
from scipy.linalg import cho_solve, cholesky

from .external import BaseModelContainer, EstimatorGlm, Model, ModelContainer, init_par
from .nbinomDeviance import nb_deviance
from .qr_decomposition import get_levenberg_start

one_millionth = 1e-6
low_value = 1e-10
supremely_low_value = 1e-13
ridiculously_low_value = 1e-100


class LevenbergEstimator:

    _train_loc: bool = False
    _train_scale: bool = False
    _model_container: BaseModelContainer

    def __init__(self, model_container: BaseModelContainer, dtype: str):
        """
        Performs initialisation and creates a new estimator.
        :param model_container:
            The model_container to be fit using Levenberg-Marquardt as in edgeR.
        :param dtype:
            i.e float64
        """
        self._model_container = model_container
        if self._model_container.design_scale.shape[1] != 1:
            raise ValueError("cannot model more than one scale parameter with edgeR/numpy backend right now.")
        self.dtype = dtype

    def train(self, maxit: int, tolerance: int = 1e-6):
        model = self._model_container
        max_x = np.max(model.x, axis=0).compute()

        n_parm = model.num_loc_params
        n_features = model.num_features

        iteration = 1

        """

        // If we start off with all entries at zero, there's really no point continuing.
        if (ymax<low_value) {
            std::fill(beta, beta+ncoefs, NA_REAL);
            std::fill(mu, mu+nlibs, 0);
            return 0;
        }

        """

        # Otherwise, we compute 'mu' based on 'beta'.

        # Iterating using reweighted least squares; setting up assorted temporary objects.

        weights = 1.0

        all_zero_features = max_x < low_value
        model.theta_location[:, all_zero_features] = np.nan
        not_done_idx = np.where(~all_zero_features)[0]
        print(not_done_idx)
        n_idx = len(not_done_idx)

        """
        For easier indexing and efficient reusage without copying arrays,
        every temporary object is set up here with the full length of n_features.
        """
        lambdas = np.zeros(n_features, dtype=float)  # step sizes, init in first iter
        levenberg_steps = np.zeros(
            n_features, dtype=int
        )  # the levenberg steps in the inner while loop, reset after each iteration
        low_deviances = np.zeros(
            n_features, dtype=bool
        )  # indicates if deviance has reached lower limit (stopping criterion)
        deviances = np.zeros(n_features, dtype=bool)  # deviance between full and fitted model
        steps = np.zeros(
            (n_parm, n_features), dtype=float
        )  # the step applied to trial update in each levenberg loop, reset after each inner iteration
        overall_steps = np.zeros_like(steps)  # collects all steps in outer
        dl = np.zeros((n_features, n_parm), dtype=float)  # the first derivative
        fim = np.zeros(
            (n_features, n_parm, n_parm), dtype=float
        )  # the fisher information matrix (used as 2nd derivative)
        fim_copy = fim.copy()  # a copy of FIM for manipulation in levenberg loop
        max_infos = np.zeros(n_features, dtype=float)  # the max value of the diagonal in the FIM
        failed_in_levenberg_loop = np.zeros(
            n_features, dtype=bool
        )  # indicates excessive damping after which it is pointless to continue
        lambda_diags = np.zeros_like(fim, dtype=float)  # a container to keep

        deviances = nb_deviance(model)

        xtwx_time = 0.0
        xtwx2_time = 0.0
        cholesky_time = 0.0
        cho_time = 0.0
        dev_time = 0.0
        lambda_time = 0.0
        rest_time = 0.0
        rest2_time = 0.0

        while n_idx > 0 and iteration <= maxit:
            print("iteration:", iteration)
            """
            Here we set up the matrix XtWX i.e. the Fisher information matrix. X is the design matrix
            and W is a diagonal matrix with the working weights for each observation (i.e. library).
            The working weights are part of the first derivative of the log-likelihood for a given coefficient,
            multiplied by any user-specified weights. When multiplied by two covariates in the design matrix,
            you get the Fisher information (i.e. variance of the log-likelihood) for that pair. This takes
            the role of the second derivative of the log-likelihood. The working weights are formed by taking
            the reciprocal of the product of the variance (in terms of the mean) and the square of the
            derivative of the link function.

            We also set up the actual derivative of the log likelihoods in 'dl'. This is done by multiplying
            each covariate by the difference between the mu and observation and dividing by the variance and
            derivative of the link function. This is then summed across all observations for each coefficient.
            The aim is to solve (XtWX)(dbeta)=dl for 'dbeta'. As XtWX is the second derivative, and dl is the
            first, you can see that we are effectively performing a multivariate Newton-Raphson procedure with
            'dbeta' as the step.
            """

            xtwx_start = time.time()
            loc = model.location_j(not_done_idx)
            scale = model.scale_j(not_done_idx)
            w = -model.fim_weight_location_location_j(not_done_idx)  # shape (obs, features)
            # print(w[:, 0].compute())
            denom = 1 + loc / scale  # shape (obs, features)
            deriv = (model.x[:, not_done_idx] - loc) / denom * weights  # shape (obs, features)
            # print((loc / denom)[:,0].compute())
            # print('deriv', deriv[:,0].compute())
            # print('denom', denom[:,0].compute())

            xh = model.xh_loc
            if isinstance(xh, dask.array.core.Array):
                xh = xh.compute()
            xhw = np.einsum("ob,of->fob", xh, w)
            fim[not_done_idx] = np.einsum("fob,oc->fbc", xhw, xh).compute()

            xtwx_time += time.time() - xtwx_start

            xtwx2_start = time.time()
            # print('fim', fim[not_done_idx[0]])
            """
            for (int lib=0; lib<nlibs; ++lib) {
                const double& cur_mu=mu[lib];
                            const double denom=(1+cur_mu*disp[lib]);
                working_weights[lib]=cur_mu/denom*w[lib];
                deriv[lib]=(y[lib]-cur_mu)/denom*w[lib];
            }

            compute_xtwx(nlibs, ncoefs, design, working_weights.data(), xtwx.data());
            """

            fim_diags = np.einsum("...ii->...i", fim[not_done_idx])  # shape (features x constrained_coefs)
            # print('shaped', deriv.shape)
            # print(model.design_loc)
            dl[not_done_idx] = np.einsum(
                "of,oc->fc", deriv, model.design_loc
            ).compute()  # shape (features, constrained_coefs)

            # print(dl[not_done_idx])

            max_infos[not_done_idx] = np.max(fim_diags, axis=1)  # shape (features,)

            # print(max_infos[0])
            # print(nb_deviance(model, [0]))

            """
            const double* dcopy=design;
            auto xtwxIt=xtwx.begin();
            for (int coef=0; coef<ncoefs; ++coef, dcopy+=nlibs, xtwxIt+=ncoefs) {
                dl[coef]=std::inner_product(deriv.begin(), deriv.end(), dcopy, 0.0);
                const double& cur_val=*(xtwxIt+coef);
                if (cur_val>max_info) { max_info=cur_val; }
            }
            """
            if iteration == 1:
                lambdas = np.maximum(max_infos * one_millionth, supremely_low_value)

            """
            Levenberg/Marquardt damping reduces step size until the deviance increases or no
            step can be found that increases the deviance. In short, increases in the deviance
            are enforced to avoid problems with convergence.
            """

            inner_idx_update = not_done_idx
            n_inner_idx = len(inner_idx_update)

            levenberg_steps.fill(0)
            failed_in_levenberg_loop[not_done_idx] = False
            f = 0
            overall_steps.fill(0)
            xtwx2_time += time.time() - xtwx2_start
            while n_inner_idx > 0:
                f += 1
                levenberg_steps[inner_idx_update] += 1
                cholesky_failed_idx = inner_idx_update.copy()  #
                cholesky_failed = np.ones(n_inner_idx, dtype=bool)
                # print('choleksy_failed', cholesky_failed)
                np.copyto(fim_copy, fim)

                m = 0
                while len(cholesky_failed_idx) > 0:
                    m += 1
                    print("cholesky_loop:", m)
                    cholesky_failed = np.zeros(len(cholesky_failed_idx), dtype=bool)
                    """
                    We need to set up copies as the decomposition routine overwrites the originals, and
                    we want the originals in case we don't like the latest step. For efficiency, we only
                    refer to the upper triangular for the XtWX copy (as it should be symmetrical). We also add
                    'lambda' to the diagonals. This reduces the step size as the second derivative is increased.
                    """
                    lambda_start = time.time()
                    lambda_diags = np.einsum(
                        "ab,bc->abc",
                        np.repeat(lambdas[cholesky_failed_idx], n_parm).reshape(len(cholesky_failed_idx), n_parm),
                        np.eye(n_parm),
                    )
                    lambda_time += time.time() - lambda_start
                    # print('lambda_diags', lambda_diags[0]);

                    fim_copy[cholesky_failed_idx] = fim[cholesky_failed_idx] + lambda_diags
                    # print(fim_copy[not_done_idx]);

                    for i, idx in enumerate(cholesky_failed_idx):
                        cholesky_start = time.time()
                        try:
                            """
                            Overwriting FIM with cholesky factorization using scipy.linalg.cholesky.
                            This is equivalent to LAPACK's dportf function (wrapper is
                            scipy.linalg.lapack.dpotrf) as used in the code from edgeR.
                            Returned is the upper triangular matrix. This is important for the steps downstream.
                            Overwriting the array is not possible here as individual slices are passed to the
                            scipy function - maybe it makes sense to use a C++ backend here and call LAPACK
                            directly as done in edgeR.
                            """
                            fim_copy[idx] = cholesky(a=fim_copy[idx], lower=False, overwrite_a=False)

                        except np.linalg.LinAlgError:
                            """
                            If it fails, it MUST mean that the matrix is singular due to numerical imprecision
                            as all the diagonal entries of the XtWX matrix must be positive. This occurs because of
                            fitted values being exactly zero; thus, the coefficients attempt to converge to negative
                            infinity. This generally forces the step size to be larger (i.e. lambda lower) in order to
                            get to infinity faster (which is impossible). Low lambda leads to numerical instability
                            and effective singularity. To solve this, we actually increase lambda; this avoids code
                            breakage to give the other coefficients a chance to converge.
                            Failure of convergence for the zero-fitted values isn't a problem as the change in
                            deviance from small --> smaller coefficients isn't that great when the true value
                            is negative inifinity.
                            """
                            lambdas[idx] *= 10
                            if lambdas[idx] <= 0:
                                lambdas[idx] = ridiculously_low_value

                            cholesky_failed[i] = True
                        cholesky_time += time.time() - cholesky_start

                    cholesky_failed_idx = cholesky_failed_idx[cholesky_failed]

                steps.fill(0)
                for i in inner_idx_update:
                    cho_start = time.time()
                    """
                    Calculating the step by solving fim_copy * step = dl using scipy.linalg.cho_solve.
                    This is equivalent to LAPACK's dpotrs function (wrapper is scipy.linalg.lapack.dpotrs)
                    as used in the code from edgeR. The int in the first argument tuple denotes lower
                    triangular (= 1) or upper triangular (= 0).
                    Again, we cannot overwrite due to a slice not passed by reference.
                    """
                    step = cho_solve((fim_copy[i], 0), dl[i], overwrite_b=False)
                    overall_steps[:, i] = step
                    steps[:, i] = step

                    cho_time += time.time() - cho_start

                # print('fim_copy', fim_copy[i])
                # print('dl', dl[i])

                # print(steps[:, 0])

                # Updating loc params.

                model.theta_location += steps

                """
                Checking if the deviance has decreased or if it's too small to care about. Either case is good
                and means that we'll be using the updated fitted values and coefficients. Otherwise, if we have
                to repeat the inner loop, then we want to do so from the original values (as we'll be scaling
                lambda up so we want to retake the step from where we were before). This is why we don't modify
                the values in-place until we're sure we want to take the step.
                """
                dev_start = time.time()

                dev_new = nb_deviance(model, inner_idx_update)  # TODO ### make this a property of model
                dev_time += time.time() - dev_start

                # print(dev_new[0])

                rest_start = time.time()

                low_deviances[inner_idx_update] = (dev_new / max_x[inner_idx_update]) < supremely_low_value

                good_updates = (dev_new <= deviances[inner_idx_update]) | low_deviances[inner_idx_update]
                idx_bad_step = inner_idx_update[~good_updates]
                rest_time = time.time() - rest_start
                # Reverse update by feature if update leads to worse loss:
                rest2_start = time.time()
                theta_location_new = model.theta_location.compute()
                rest2_time += time.time() - rest2_start

                theta_location_new[:, idx_bad_step] = theta_location_new[:, idx_bad_step] - steps[:, idx_bad_step]
                model.theta_location = theta_location_new
                good_idx = inner_idx_update[good_updates]
                if len(good_idx) > 0:
                    deviances[good_idx] = dev_new[good_updates]

                # Increasing lambda, to increase damping. Again, we have to make sure it's not zero.
                lambdas[idx_bad_step] = np.where(
                    lambdas[idx_bad_step] <= 0, ridiculously_low_value, lambdas[idx_bad_step] * 2
                )

                # Excessive damping; steps get so small that it's pointless to continue.
                failed_in_levenberg_loop[inner_idx_update] = (
                    lambdas[inner_idx_update] / max_infos[inner_idx_update]
                ) > (1 / supremely_low_value)

                inner_idx_update = inner_idx_update[
                    ~(good_updates | failed_in_levenberg_loop[inner_idx_update])
                ]  # the features for which both the update was reversed and the step size is not too small yet

                n_inner_idx = len(inner_idx_update)

                # print('n_inner', inner_idx_update)

            """
            Terminating if we failed, if divergence from the exact solution is acceptably low
            (cross-product of dbeta with the log-likelihood derivative) or if the actual deviance
            of the fit is acceptably low.
            """
            divergence = np.einsum("fc,cf->f", dl[not_done_idx], overall_steps[:, not_done_idx])
            not_done_idx = not_done_idx[
                (divergence >= tolerance) & ~low_deviances[not_done_idx] & ~failed_in_levenberg_loop[not_done_idx]
            ]

            n_idx = len(not_done_idx)
            """
            If we quit the inner levenberg loop immediately and survived all the break conditions above,
            that means that deviance is decreasing substantially. Thus, we need larger steps to get there faster.
            To do so, we decrease the damping factor. Note that this only applies if we didn't decrease the
            damping factor in the inner levenberg loop, as that would indicate that we need to slow down.
            """

            lambdas[levenberg_steps == 1] /= 10
            iteration += 1
            # print('..............................', model.theta_location[:,not_done_idx].compute())

        return xtwx_time, xtwx2_time, lambda_time, cholesky_time, cho_time, dev_time, rest_time, rest2_time


class NBEstimator(LevenbergEstimator):
    def __init__(
        self,
        model: Model,
        dispersion: float,
        dtype: str = "float64",
    ):
        """
        Performs initialisation using QR decomposition as in edgeR and creates a new estimator.

        :param model: The NBModel object to fit.
        :param dispersion: The fixed dispersion parameter to use during fitting the loc model.
        :param dtype: Numerical precision.
        """
        init_theta_location = get_levenberg_start(model=model, disp=dispersion, use_null=True)
        init_theta_scale = np.full((1, model.num_features), np.log(1 / dispersion))
        self._train_loc = True
        self._train_scale = False  # This is fixed as edgeR doesn't fit the scale parameter

        _model_container = ModelContainer(
            model=model,
            init_theta_location=init_theta_location,
            init_theta_scale=init_theta_scale,
            chunk_size_genes=model.chunk_size_genes,
            dtype=dtype,
        )
        super(NBEstimator, self).__init__(model_container=_model_container, dtype=dtype)

    # def init_par(model, init_location):
