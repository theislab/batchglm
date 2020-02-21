import abc
import dask.array
import logging
import multiprocessing
import numpy as np
import pprint
import scipy
import scipy.sparse
import scipy.optimize
import sparse
import sys
import time
from typing import Tuple

from .external import _EstimatorGLM, pkg_constants
from .training_strategies import TrainingStrategies

logger = logging.getLogger("batchglm")


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
        if input_data.design_scale.shape[1] != 1:
            raise ValueError("cannot model more than one scale parameter with numpy backend right now.")
        _EstimatorGLM.__init__(
            self=self,
            model=model,
            input_data=input_data
        )
        self.dtype = dtype
        self.values = []
        self.lls = []

        self.TrainingStrategies = TrainingStrategies

    def initialize(self):
        pass

    def train(
            self,
            max_steps: int = 100,
            method_b: str = "brent",
            update_b_freq: int = 5,
            ftol_b: float = 1e-8,
            lr_b: float = 1e-2,
            max_iter_b: int = 1000,
            nproc: int = 3,
            **kwargs
    ):
        """
        Train GLM.

        Convergence decision:
        Location and scale model updates are done in separate iterations and are done with different algorithms.
        Scale model updates are much less frequent (only every update_b_freq-th iteration) as they are much slower.
        During a stretch of update_b_freq number of location model updates between two scale model updates, convergence
        of the location model is tracked with self.model.converged. This is re-set after a scale model update, as this
        convergence only holds conditioned on a particular scale model value.
        Full convergence of a feature wise model is evaluated after each scale model update: If the loss function based
        convergence criterium holds across the cumulative updates of the sequence of location updates and last scale
        model update, the feature is considered converged. For this, the loss value at the last scale model update is
        save in ll_last_b_update. Full convergence is saved in fully_converged.

        :param max_steps:
        :param method_b:
        :param update_b_freq: One over minimum frequency of scale model updates per location model update.
            A scale model update will be run at least every update_b_freq number of location model update iterations.
        :param ftol_b:
        :param lr_b:
        :param max_iter_b:
        :param nproc:
        :param kwargs:
        :return:
        """
        # Iterate until conditions are fulfilled.
        train_step = 0
        epochs_until_b_update = update_b_freq
        fully_converged = np.tile(False, self.model.model_vars.n_features)

        ll_current = - self.model.ll_byfeature.compute()
        ll_last_b_update = ll_current.copy()
        #logging.getLogger("batchglm").info(
        sys.stdout.write("iter   %i: ll=%f\n" % (0, np.sum(ll_current)))
        while np.any(np.logical_not(fully_converged)) and \
                train_step < max_steps:
            t0 = time.time()
            # Line search step for scale model:
            # Run this update every update_b_freq iterations.
            if epochs_until_b_update == 0:
                # Compute update.
                idx_update = np.where(np.logical_not(fully_converged))[0]
                b_step = self.b_step(
                    idx_update=idx_update,
                    method=method_b,
                    ftol=ftol_b,
                    lr=lr_b,
                    max_iter=max_iter_b,
                    nproc=nproc
                )
                # Perform trial update.
                self.model.b_var = self.model.b_var + b_step
                # Reverse update by feature if update leads to worse loss:
                ll_proposal = - self.model.ll_byfeature_j(j=idx_update).compute()
                idx_bad_step = idx_update[np.where(ll_proposal > ll_current[idx_update])[0]]
                if isinstance(self.model.b_var, dask.array.core.Array):
                    b_var_new = self.model.b_var.compute()
                else:
                    b_var_new = self.model.b_var.copy()
                b_var_new[:, idx_bad_step] = b_var_new[:, idx_bad_step] - b_step[:, idx_bad_step]
                self.model.b_var = b_var_new
                # Update likelihood vector with updated genes based on already evaluated proposal likelihood.
                ll_new = ll_current.copy()
                ll_new[idx_update] = ll_proposal
                ll_new[idx_bad_step] = ll_current[idx_bad_step]
                # Reset b model update counter.
                epochs_until_b_update = update_b_freq
            else:
                # IWLS step for location model:
                # Compute update.
                idx_update = self.model.idx_not_converged
                a_step = self.iwls_step(idx_update=idx_update)
                # Perform trial update.
                self.model.a_var = self.model.a_var + a_step
                # Reverse update by feature if update leads to worse loss:
                ll_proposal = - self.model.ll_byfeature_j(j=idx_update).compute()
                idx_bad_step = idx_update[np.where(ll_proposal > ll_current[idx_update])[0]]
                if isinstance(self.model.b_var, dask.array.core.Array):
                    a_var_new = self.model.a_var.compute()
                else:
                    a_var_new = self.model.a_var.copy()
                a_var_new[:, idx_bad_step] = a_var_new[:, idx_bad_step] - a_step[:, idx_bad_step]
                self.model.a_var = a_var_new
                # Update likelihood vector with updated genes based on already evaluated proposal likelihood.
                ll_new = ll_current.copy()
                ll_new[idx_update] = ll_proposal
                ll_new[idx_bad_step] = ll_current[idx_bad_step]
                # Update epoch counter of a updates until next b update:
                epochs_until_b_update -= 1

            # Evaluate and update convergence:
            ll_previous = ll_current
            ll_current = ll_new
            if epochs_until_b_update == update_b_freq:  # b step update was executed.
                # Update terminal convergence in fully_converged and intermediate convergence in self.model.converged.
                converged_f = np.logical_or(
                    ll_last_b_update < ll_current,  # loss gets worse
                    np.abs(ll_last_b_update - ll_current) / np.maximum(  # relative decrease in loss is too small
                        np.nextafter(0, np.inf, dtype=ll_previous.dtype),  # catch division by zero
                        np.abs(ll_last_b_update)
                    ) < pkg_constants.LLTOL_BY_FEATURE,
                )
                self.model.converged = np.logical_or(fully_converged, converged_f)
                ll_last_b_update = ll_current.copy()
                fully_converged = self.model.converged.copy()
            else:
                # Update intermediate convergence in self.model.converged.
                converged_f = np.logical_or(
                    ll_previous < ll_current,  # loss gets worse
                    np.abs(ll_previous - ll_current) / np.maximum(  # relative decrease in loss is too small
                        np.nextafter(0, np.inf, dtype=ll_previous.dtype),  # catch division by zero
                        np.abs(ll_previous)
                    ) < pkg_constants.LLTOL_BY_FEATURE,
                )
                self.model.converged = np.logical_or(self.model.converged, converged_f)
                if np.all(self.model.converged):
                    # All location models are converged. This means that the next update will be b model
                    # update and all remaining intermediate a model updates can be skipped:
                    epochs_until_b_update = 0

            # Conclude and report iteration.
            train_step += 1
            #logging.getLogger("batchglm").info(
            sys.stdout.write(
                "iter %s: ll=%f, converged: %.2f%% (loc: %.2f%%, scale update: %s), in %.2fsec\n" %
                (
                    (" " if train_step < 10 else "") + (" " if train_step < 100 else "") + str(train_step),
                    np.sum(ll_current),
                    np.mean(fully_converged)*100,
                    np.mean(self.model.converged) * 100,
                    str(epochs_until_b_update == update_b_freq),
                    time.time()-t0
                )
            )
            #sys.stdout.write(
            #    '\riter %i: ll=%f, %.2f%% converged' %
            #    (train_step, np.sum(ll_current), np.round(np.mean(delayed_converged)*100, 2))
            #)
            #sys.stdout.flush()
            self.lls.append(ll_current)
        #sys.stdout.write('\r')
        #sys.stdout.flush()

    def a_step_gd(
            self,
            idx: np.ndarray,
            ftol: float,
            max_iter: int,
            lr: float
    ) -> np.ndarray:
        """
        Not used

        :return:
        """
        iter = 0
        a_var_old = self.model.a_var.compute()
        converged = np.tile(True, self.model.model_vars.n_features)
        converged[idx] = False
        ll_current = - self.model.ll_byfeature.compute()
        while np.any(np.logical_not(converged)) and iter < max_iter:
            idx_to_update = np.where(np.logical_not(converged))[0]
            jac = np.zeros_like(self.model.a_var).compute()
            # Use mean jacobian so that learning rate is independent of number of samples.
            jac[:, idx_to_update] = - self.model.jac_a.compute().T[:, idx_to_update] / \
                                    self.model.input_data.num_observations
            self.model._a_var = self.model.a_var.compute() + lr * jac
            # Assess convergence:
            ll_previous = ll_current
            ll_current = - self.model.ll_byfeature.compute()
            converged_f = (ll_current - ll_previous) / ll_previous > -ftol
            a_var_new = self.model.a_var.compute()
            a_var_new[:, converged_f] = a_var_new[:, converged_f] - lr * jac[:, converged_f]
            self.model.a_var = a_var_new
            converged = np.logical_or(converged, converged_f)
            iter += 1
            logging.getLogger("batchglm").info(
                "iter %i: ll=%f, converged location model: %.2f%%" %
                (
                    (" " if iter < 10 else "") + (" " if iter < 100 else "") + str(iter),
                    np.sum(ll_current),
                    np.mean(converged) * 100
                )
            )
        return self.model.a_var.compute() - a_var_old

    def iwls_step(
            self,
            idx_update: np.ndarray
    ) -> np.ndarray:
        """

        :return: (inferred param x features)
        """
        w = self.model.fim_weight_aa_j(j=idx_update)  # (observations x features)
        ybar = self.model.ybar_j(j=idx_update)  # (observations x features)
        # Translate to problem of form ax = b for each feature:
        # (in the following, X=design and Y=counts)
        # a=X^T*W*X: ([features] x inferred param)
        # x=theta: ([features] x inferred param)
        # b=X^T*W*Ybar: ([features] x inferred param)
        xh = np.matmul(self.model.design_loc, self.model.constraints_loc)
        xhw = np.einsum('ob,of->fob', xh, w)
        a = np.einsum('fob,oc->fbc', xhw, xh)
        b = np.einsum('fob,of->fb', xhw, ybar)

        delta_theta = np.zeros_like(self.model.a_var)
        if isinstance(delta_theta, dask.array.core.Array):
            delta_theta = delta_theta.compute()

        if isinstance(a, dask.array.core.Array):
            # Have to use a workaround to solve problems in parallel in dask here. This workaround does
            # not work if there is only a single problem, ie. if the first dimension of a and b has length 1.
            if a.shape[0] != 1:
                delta_theta[:, idx_update] = dask.array.map_blocks(
                    np.linalg.solve, a, b[:, :, None], chunks=b[:, :, None].shape
                ).squeeze().T.compute()
            else:
                delta_theta[:, idx_update] = np.expand_dims(
                    np.linalg.solve(a[0], b[0]).compute(),
                    axis=-1
                )
        else:
            delta_theta[:, idx_update] = np.linalg.solve(a, b).T
        # Via np.linalg.lsts:
        #delta_theta[:, idx_update] = np.concatenate([
        #    np.expand_dims(np.linalg.lstsq(a[i, :, :], b[i, :])[0], axis=-1)
        #    for i in idx_update)
        #], axis=-1)
        # Via np.linalg.inv:
        # #delta_theta[:, idx_update] = np.concatenate([
        #    np.expand_dims(np.matmul(np.linalg.inv(a[i, :, :]), b[i, :]), axis=-1)
        #    for i in idx_update)
        #], axis=-1)
        return delta_theta

    def b_step(
            self,
            idx_update: np.ndarray,
            method: str,
            ftol: float,
            lr: float,
            max_iter: int,
            nproc: int
    ) -> np.ndarray:
        """

        :return:
        """
        if method.lower() in ["gd"]:
            return self._b_step_gd(
                idx_update=idx_update,
                ftol=ftol,
                lr=lr,
                max_iter=max_iter
            )
        else:
            return self._b_step_loop(
                idx_update=idx_update,
                method=method,
                ftol=ftol,
                max_iter=max_iter,
                nproc=nproc
            )

    def _b_step_gd(
            self,
            idx_update: np.ndarray,
            ftol: float,
            max_iter: int,
            lr: float
    ) -> np.ndarray:
        """

        :return:
        """
        iter = 0
        b_var_old = self.model.b_var.compute()
        converged = np.tile(True, self.model.model_vars.n_features)
        converged[idx_update] = False
        ll_current = - self.model.ll_byfeature.compute()
        while np.any(np.logical_not(converged)) and iter < max_iter:
            idx_to_update = np.where(np.logical_not(converged))[0]
            jac = np.zeros_like(self.model.b_var).compute()
            # Use mean jacobian so that learning rate is independent of number of samples.
            jac[:, idx_to_update] = self.model.jac_b_j(j=idx_to_update).compute().T / \
                                    self.model.input_data.num_observations
            self.model.b_var_j_setter(
                value=(self.model.b_var.compute() + lr * jac)[:, idx_to_update],
                j=idx_to_update
            )
            # Assess convergence:
            ll_previous = ll_current
            ll_current = - self.model.ll_byfeature.compute()
            converged_f = (ll_current - ll_previous) / ll_previous > -ftol
            b_var_new = self.model.b_var.compute()
            b_var_new[:, converged_f] = b_var_new[:, converged_f] - lr * jac[:, converged_f]
            self.model.b_var = b_var_new
            converged = np.logical_or(converged, converged_f)
            iter += 1
            logging.getLogger("batchglm").info(
                "iter %i: ll=%f, converged scale model: %.2f%%" %
                (
                    (" " if iter < 10 else "") + (" " if iter < 100 else "") + str(iter),
                    np.sum(ll_current),
                    np.mean(converged) * 100
                )
            )
        return self.model.b_var.compute() - b_var_old

    def optim_handle(
            self,
            b_j,
            data_j,
            eta_loc_j,
            xh_scale,
            max_iter,
            ftol
    ):
        # Need to supply dense numpy array to scipy optimize:
        if isinstance(data_j, sparse.COO) or isinstance(data_j, scipy.sparse.csr_matrix):
            data_j = data_j.todense()
        if len(data_j.shape) == 1:
            data_j = np.expand_dims(data_j, axis=-1)

        ll = self.model.ll_handle()
        lb, ub = self.model.param_bounds(dtype=data_j.dtype)
        lb_bracket = np.max([lb["b_var"], b_j - 20])
        ub_bracket = np.min([ub["b_var"], b_j + 20])

        def cost_b_var(x, data_jj, eta_loc_jj, xh_scale_jj):
            x = np.clip(np.array([[x]]), lb["b_var"], ub["b_var"])
            return - np.sum(ll(data_jj, eta_loc_jj, x, xh_scale_jj))

        # jac_b = self.model.jac_b_handle()
        # def cost_b_var_prime(x, data_jj, eta_loc_jj, xh_scale_jj):
        #    x = np.clip(np.array([[x]]), lb["b_var"], ub["b_var"])
        #    return - np.sum(jac_b(data_jj, eta_loc_jj, x, xh_scale_jj))
        # return scipy.optimize.line_search(
        #    f=cost_b_var,
        #    myfprime=cost_b_var_prime,
        #    args=(data_j, eta_loc_j, xh_scale),
        #    maxiter=max_iter,
        #    xk=b_j+5,
        #    pk=-np.ones_like(b_j)
        # )

        return scipy.optimize.brent(
            func=cost_b_var,
            args=(data_j, eta_loc_j, xh_scale),
            maxiter=max_iter,
            tol=ftol,
            brack=(lb_bracket, ub_bracket),
            full_output=True
        )

    def _b_step_loop(
            self,
            idx_update: np.ndarray,
            method: str,
            max_iter: int,
            ftol: float,
            nproc: int
    ) -> np.ndarray:
        """

        :return:
        """
        delta_theta = np.zeros_like(self.model.b_var)
        if isinstance(delta_theta, dask.array.core.Array):
            delta_theta = delta_theta.compute()

        xh_scale = np.matmul(self.model.design_scale, self.model.constraints_scale).compute()
        b_var = self.model.b_var.compute()
        if nproc > 1 and len(idx_update) > nproc:
            sys.stdout.write('\rFitting %i dispersion models: (progress not available with multiprocessing)' % len(idx_update))
            sys.stdout.flush()
            with multiprocessing.Pool(processes=nproc) as pool:
                x = self.x.compute()
                eta_loc = self.model.eta_loc.compute()
                results = pool.starmap(
                    self.optim_handle,
                    [(
                        b_var[0, j],
                        x[:, [j]],
                        eta_loc[:, [j]],
                        xh_scale,
                        max_iter,
                        ftol
                    ) for j in idx_update]
                )
                pool.close()
            delta_theta[0, idx_update] = np.array([x[0] for x in results])
            sys.stdout.write('\r')
            sys.stdout.flush()
        else:
            t0 = time.time()
            for i, j in enumerate(idx_update):
                sys.stdout.write(
                    '\rFitting dispersion models: %.2f%% in %.2fsec' %
                    (
                        np.round(i / len(idx_update) * 100., 2),
                        time.time() - t0
                    )
                )
                sys.stdout.flush()
                if method.lower() == "brent":
                    eta_loc = self.model.eta_loc_j(j=j).compute()
                    data = self.x[:, [j]].compute()
                    # Need to supply dense numpy array to scipy optimize:
                    if isinstance(data, sparse.COO) or isinstance(data, scipy.sparse.csr_matrix):
                        data = data.todense()

                    ll = self.model.ll_handle()
                    lb, ub = self.model.param_bounds(dtype=data.dtype)
                    lb_bracket = np.max([lb["b_var"], b_var[0, j] - 20])
                    ub_bracket = np.min([ub["b_var"], b_var[0, j] + 20])

                    def cost_b_var(x, data_j, eta_loc_j, xh_scale_j):
                        x = np.clip(np.array([[x]]), lb["b_var"], ub["b_var"])
                        return - np.sum(ll(
                            data_j,
                            eta_loc_j,
                            x,
                            xh_scale_j
                        ))

                    delta_theta[0, j] = scipy.optimize.brent(
                        func=cost_b_var,
                        args=(data, eta_loc, xh_scale),
                        maxiter=max_iter,
                        tol=ftol,
                        brack=(lb_bracket, ub_bracket),
                        full_output=False
                    )
                else:
                    raise ValueError("method %s not recognized" % method)
            sys.stdout.write('\r')
            sys.stdout.flush()

        if isinstance(self.model.b_var, dask.array.core.Array):
            delta_theta[:, idx_update] = delta_theta[:, idx_update] - self.model.b_var.compute()[:, idx_update]
        else:
            delta_theta[:, idx_update] = delta_theta[:, idx_update] - self.model.b_var.copy()[:, idx_update]
        return delta_theta

    def finalize(self):
        """
        Evaluate all tensors that need to be exported from session and save these as class attributes
        and close session.

        Changes .model entry from tf1-based EstimatorGraph to numpy based Model instance and
        transfers relevant attributes.
        """
        # Read from numpy-IRLS estimator specific model:
        self._hessian = - self.model.fim.compute()
        self._fisher_inv = np.linalg.inv(- self._hessian)
        self._jacobian = np.sum(np.abs(self.model.jac.compute() / self.model.x.shape[0]), axis=1)
        self._log_likelihood = self.model.ll_byfeature.compute()
        self._loss = np.sum(self._log_likelihood)

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

