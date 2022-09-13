import sys
from typing import Union

import dask.array
import numpy as np
from scipy.linalg import cho_solve, cholesky

from .c_utils import nb_deviance
from .external import BaseModelContainer, EstimatorGlm, InputDataGLM, ModelContainer, NBModel, init_par
from .glm_one_group import fit_single_group, get_single_group_start
from .qr_decomposition import get_levenberg_start

one_millionth = 1e-6
low_value = 1e-10
supremely_low_value = 1e-13
ridiculously_low_value = 1e-100


class Estimator:

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

        """
        check which algorithm to use. We can use a shortcut algorithm if the number of unique rows in the design
        matrix is equal to the number of coefficients.
        """
        if isinstance(self._model_container.design_loc, dask.array.core.Array):
            unique_design = np.unique(self._model_container.design_loc.compute(), axis=0)
        else:
            unique_design = np.unique(self._model_container.design_loc, axis=0)

        if unique_design.shape[0] == unique_design.shape[1]:
            self.fitting_algorithm = "one_way"
        else:
            self.fitting_algorithm = "levenberg"
            self._model_container.theta_location = get_levenberg_start(
                model=self._model_container.model, disp=self._model_container.scale, use_null=True
            )

    def train(self, maxit: int, tolerance: float = 1e-6):

        if self.fitting_algorithm == "one_way":
            self.train_oneway(maxit=maxit, tolerance=tolerance)
        elif self.fitting_algorithm == "levenberg":
            self.train_levenberg(maxit=maxit, tolerance=tolerance)
        else:
            raise ValueError(f"Unrecognized algorithm: {self.train_levenberg}")

    def train_oneway(self, maxit: int, tolerance: float):
        model = self._model_container
        if isinstance(model.design_loc, dask.array.core.Array):
            unique_design, group_idx = np.unique(model.design_loc.compute(), return_inverse=True, axis=0)
        else:
            unique_design, group_idx = np.unique(model.design_loc, return_inverse=True, axis=0)

        n_groups = unique_design.shape[1]

        theta_location = model.theta_location
        if isinstance(theta_location, dask.array.core.Array):
            theta_location = theta_location.compute()  # .copy()

        for i in range(n_groups):
            obs_group = np.where(group_idx == i)[0]
            dloc = model.design_loc
            if isinstance(model.design_loc, dask.array.core.Array):
                dloc = dloc.compute()
            sf = model.size_factors
            if sf is not None:
                sf = sf[obs_group]
                if isinstance(model.size_factors, dask.array.core.Array):
                    sf = sf.compute()
            dscale = model.design_scale
            if isinstance(model.design_loc, dask.array.core.Array):
                dscale = dscale.compute()
            group_model = model.model.__class__(
                InputDataGLM(
                    data=model.x[obs_group],
                    design_loc=dloc[np.ix_(obs_group, np.ndarray([i]))],
                    design_loc_names=model.design_loc_names[[i]],
                    size_factors=sf,
                    design_scale=dscale[np.ix_(obs_group, np.ndarray([0]))],
                    design_scale_names=model.design_scale_names[[0]],
                    as_dask=isinstance(model.x, dask.array.core.Array),
                    chunk_size_cells=model.chunk_size_cells,
                    chunk_size_genes=model.chunk_size_genes,
                )
            )
            group_model = ModelContainer(
                model=group_model,
                init_theta_location=get_single_group_start(group_model.x, group_model.size_factors),
                init_theta_scale=model.theta_scale,
                chunk_size_genes=model.chunk_size_genes,
                dtype=model.theta_location.dtype,
            )
            fit_single_group(group_model, maxit=maxit, tolerance=tolerance)
            if isinstance(group_model.theta_location, dask.array.core.Array):
                theta_location[i] = group_model.theta_location.compute()
            else:
                theta_location[i] = group_model.theta_location

        theta_location = np.linalg.solve(unique_design, theta_location)
        model.theta_location = theta_location

    def train_levenberg(self, maxit: int, tolerance: float = 1e-6):
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
            loc = model.location_j(not_done_idx)
            scale = model.scale_j(not_done_idx)
            w = -model.fim_weight_location_location_j(not_done_idx)  # shape (obs, features)
            denom = 1 + loc / scale  # shape (obs, features)
            deriv = (model.x[:, not_done_idx] - loc) / denom * weights  # shape (obs, features)
            xh = model.xh_loc

            xhw = np.einsum("ob,of->fob", xh, w)
            fim[not_done_idx] = np.einsum("fob,oc->fbc", xhw, xh)  # .compute()

            fim_diags = np.einsum("...ii->...i", fim[not_done_idx])  # shape (features x constrained_coefs)

            dl[not_done_idx] = np.einsum("of,oc->fc", deriv, model.design_loc)

            max_infos[not_done_idx] = np.max(fim_diags, axis=1)  # shape (features,)

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
            while n_inner_idx > 0:
                f += 1
                levenberg_steps[inner_idx_update] += 1
                cholesky_failed_idx = inner_idx_update.copy()  #
                cholesky_failed = np.ones(n_inner_idx, dtype=bool)
                np.copyto(fim_copy, fim)

                m = 0
                while len(cholesky_failed_idx) > 0:
                    m += 1
                    cholesky_failed = np.zeros(len(cholesky_failed_idx), dtype=bool)
                    """
                    We need to set up copies as the decomposition routine overwrites the originals, and
                    we want the originals in case we don't like the latest step. For efficiency, we only
                    refer to the upper triangular for the XtWX copy (as it should be symmetrical). We also add
                    'lambda' to the diagonals. This reduces the step size as the second derivative is increased.
                    """

                    lambda_diags = np.einsum(
                        "ab,bc->abc",
                        np.repeat(lambdas[cholesky_failed_idx], n_parm).reshape(len(cholesky_failed_idx), n_parm),
                        np.eye(n_parm),
                    )

                    fim_copy[cholesky_failed_idx] = fim[cholesky_failed_idx] + lambda_diags

                    for i, idx in enumerate(cholesky_failed_idx):
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

                    cholesky_failed_idx = cholesky_failed_idx[cholesky_failed]

                steps.fill(0)
                for i in inner_idx_update:
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

                # Updating loc params.

                model.theta_location += steps

                """
                Checking if the deviance has decreased or if it's too small to care about. Either case is good
                and means that we'll be using the updated fitted values and coefficients. Otherwise, if we have
                to repeat the inner loop, then we want to do so from the original values (as we'll be scaling
                lambda up so we want to retake the step from where we were before). This is why we don't modify
                the values in-place until we're sure we want to take the step.
                """

                dev_new = nb_deviance(model, inner_idx_update)  # TODO ### make this a property of model

                low_deviances[inner_idx_update] = (dev_new / max_x[inner_idx_update]) < supremely_low_value

                good_updates = (dev_new <= deviances[inner_idx_update]) | low_deviances[inner_idx_update]
                idx_bad_step = inner_idx_update[~good_updates]

                # Reverse update by feature if update leads to worse loss:
                theta_location_new = model.theta_location.compute()

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

    def reset_theta_scale(self, new_scale: Union[np.ndarray, dask.array.core.Array, float]):
        if isinstance(new_scale, float):
            new_scale = np.full(self._model_container.theta_scale.shape, new_scale)
        self._model_container.theta_scale = new_scale


class NBEstimator(Estimator):
    def __init__(
        self,
        model: NBModel,
        dispersion: float,
        dtype: str = "float64",
    ):
        """
        Performs initialisation using QR decomposition as in edgeR and creates a new estimator.

        :param model: The NBModel object to fit.
        :param dispersion: The fixed dispersion parameter to use during fitting the loc model.
        :param dtype: Numerical precision.
        """
        init_theta_location = np.zeros((model.xh_loc.shape[1], model.num_features), dtype=model.cast_dtype)
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
