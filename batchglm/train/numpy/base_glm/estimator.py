import abc
import logging
import multiprocessing
import sys
import time
from typing import Tuple, List

import dask.array
import numpy as np
import scipy
import scipy.optimize
import scipy.sparse
import sparse
from enum import Enum
import pprint


from .external import pkg_constants
from .training_strategies import TrainingStrategies

from .modelContainer import BaseModelContainer

logger = logging.getLogger("batchglm")


class EstimatorGlm(metaclass=abc.ABCMeta):
    """
    Estimator for Generalized Linear Models (GLMs).

    Attributes
    ----------
    dtype : str
    lls : List
        A list of all log likelihood updates
    """

    _train_loc: bool = False
    _train_scale: bool = False
    _modelContainer: BaseModelContainer
    lls: List[float] = []
    dtype: str = ''

    def __init__(
        self,
        dtype: str,
        provide_batched: bool = False

    ):
        """
        Performs initialisation and creates a new estimator.
        :param model:
            The IWLS model to be fit
        :param dtype:
            i.e float64
        """
        if self.modelContainer.design_scale.shape[1] != 1:
            raise ValueError("cannot model more than one scale parameter with numpy backend right now.")
        # _EstimatorGLM.__init__(self=self, model=model)
        self.dtype = dtype
        self.lls = []

        self.TrainingStrategies = TrainingStrategies

    @property
    def train_loc(self) -> bool:
        return self._train_loc

    @property
    def train_scale(self) -> bool:
        return self._train_scale

    @property
    def modelContainer(self) -> BaseModelContainer:
        return self._modelContainer

    def train_sequence(self, training_strategy, **kwargs):
        if isinstance(training_strategy, Enum):
            training_strategy = training_strategy.value
        elif isinstance(training_strategy, str):
            training_strategy = self.TrainingStrategies[training_strategy].value

        if training_strategy is None:
            training_strategy = self.TrainingStrategies.DEFAULT.value

        logger.debug("training strategy:\n%s", pprint.pformat(training_strategy))
        for idx, d in enumerate(training_strategy):
            logger.debug("Beginning with training sequence #%d", idx + 1)
            # Override duplicate arguments with user choice:
            if np.any([x in list(d.keys()) for x in list(kwargs.keys())]):
                d = dict([(x, y) for x, y in d.items() if x not in list(kwargs.keys())])
                for x in [xx for xx in list(d.keys()) if xx in list(kwargs.keys())]:
                    sys.stdout.write(
                        "overrding %s from training strategy with value %s with new value %s\n"
                        % (x, str(d[x]), str(kwargs[x]))
                    )
            self.train(**d, **kwargs)
            logger.debug("Training sequence #%d complete", idx + 1)

    def initialize(self):
        pass

    def train(
        self,
        max_steps: int = 100,
        method_scale: str = "brent",
        update_scale_freq: int = 5,
        ftol_scale: float = 1e-8,
        lr_scale: float = 1e-2,
        max_iter_scale: int = 1000,
        nproc: int = 3,
        **kwargs
    ):
        """
        Train GLM.

        Convergence decision:
        Location and scale model updates are done in separate iterations and are done with different algorithms.
        Scale model updates are much less frequent (only every update_scale_freq-th iteration) as they are much slower.
        During a stretch of update_scale_freq number of location model updates between two scale model updates,
        convergence of the location model is tracked with self.model.converged.
        This is re-set after a scale model update,
        as this convergence only holds conditioned on a particular scale model value.
        Full convergence of a feature wise model is evaluated after each scale model update: If the loss function based
        convergence criterium holds across the cumulative updates of the sequence of location updates and last scale
        model update, the feature is considered converged. For this, the loss value at the last scale model update is
        save in ll_last_scale_update. Full convergence is saved in fully_converged.

        :param max_steps:
        :param method_scale:
        :param update_scale_freq: One over minimum frequency of scale model updates per location model update.
            A scale model update will be run at least
            every update_scale_freq number of location model update iterations.
        :param ftol_scale:
        :param lr_scale:
        :param max_iter_scale:
        :param nproc:
        :param kwargs:
        :return:
        """
        # Iterate until conditions are fulfilled.
        train_step = 0
        if self._train_scale:
            if not self._train_loc:
                update_scale_freq = 1
        else:
            update_scale_freq = int(1e10)  # very large integer
        epochs_until_scale_update = update_scale_freq
        fully_converged = np.tile(False, self.modelContainer.num_features)

        ll_current = -self.modelContainer.ll_byfeature.compute()
        ll_last_scale_update = ll_current.copy()
        # logging.getLogger("batchglm").info(
        sys.stdout.write("iter   %i: ll=%f\n" % (0, np.sum(ll_current)))
        while np.any(np.logical_not(fully_converged)) and train_step < max_steps:
            t0 = time.time()
            # Line search step for scale model:
            # Run this update every update_scale_freq iterations.
            if epochs_until_scale_update == 0:
                # Compute update.
                idx_update = np.where(np.logical_not(fully_converged))[0]
                if self._train_scale:
                    b_step = self.b_step(
                        idx_update=idx_update,
                        method=method_scale,
                        ftol=ftol_scale,
                        lr=lr_scale,
                        max_iter=max_iter_scale,
                        nproc=nproc,
                    )
                    # Perform trial update.
                    self.modelContainer.theta_scale = self.modelContainer.theta_scale + b_step
                    # Reverse update by feature if update leads to worse loss:
                    ll_proposal = -self.modelContainer.ll_byfeature_j(j=idx_update).compute()
                    idx_bad_step = idx_update[np.where(ll_proposal > ll_current[idx_update])[0]]
                    if isinstance(self.modelContainer.theta_scale, dask.array.core.Array):
                        theta_scale_new = self.modelContainer.theta_scale.compute()
                    else:
                        theta_scale_new = self.modelContainer.theta_scale.copy()
                    theta_scale_new[:, idx_bad_step] = theta_scale_new[:, idx_bad_step] - b_step[:, idx_bad_step]
                    self.modelContainer.theta_scale = theta_scale_new
                else:
                    ll_proposal = ll_current[idx_update]
                    idx_bad_step = np.array([], dtype=np.int32)
                # Update likelihood vector with updated genes based on already evaluated proposal likelihood.
                ll_new = ll_current.copy()
                ll_new[idx_update] = ll_proposal
                ll_new[idx_bad_step] = ll_current[idx_bad_step]
                # Reset b model update counter.
                epochs_until_scale_update = update_scale_freq
            else:
                # IWLS step for location model:
                # Compute update.
                idx_update = self.modelContainer.idx_not_converged
                if self._train_loc:
                    a_step = self.iwls_step(idx_update=idx_update)
                    # Perform trial update.
                    self.modelContainer.theta_location += a_step
                    # Reverse update by feature if update leads to worse loss:
                    ll_proposal = -self.modelContainer.ll_byfeature_j(j=idx_update).compute()
                    idx_bad_step = idx_update[np.where(ll_proposal > ll_current[idx_update])[0]]
                    if isinstance(self.modelContainer.theta_location, dask.array.core.Array):
                        theta_location_new = self.modelContainer.theta_location.compute()
                    else:
                        theta_location_new = self.modelContainer.theta_location.copy()
                    theta_location_new[:, idx_bad_step] = theta_location_new[:, idx_bad_step] - a_step[:, idx_bad_step]
                    self.modelContainer.theta_location = theta_location_new
                else:
                    ll_proposal = ll_current[idx_update]
                    idx_bad_step = np.array([], dtype=np.int32)
                # Update likelihood vector with updated genes based on already evaluated proposal likelihood.
                ll_new = ll_current.copy()
                ll_new[idx_update] = ll_proposal
                ll_new[idx_bad_step] = ll_current[idx_bad_step]
                # Update epoch counter of a updates until next b update:
                epochs_until_scale_update -= 1

            # Evaluate and update convergence:
            
            ll_previous = ll_current
            ll_current = ll_new
            if epochs_until_scale_update == update_scale_freq:  # b step update was executed.
                # Update terminal convergence in fully_converged and intermediate convergence in self.model.converged.
                converged_f = np.logical_or(
                    ll_last_scale_update < ll_current,  # loss gets worse
                    np.abs(ll_last_scale_update - ll_current)
                    / np.maximum(  # relative decrease in loss is too small
                        np.nextafter(0, np.inf, dtype=ll_previous.dtype),  # catch division by zero
                        np.abs(ll_last_scale_update),
                    )
                    < pkg_constants.LLTOL_BY_FEATURE,
                )
                self.modelContainer.converged = np.logical_or(fully_converged, converged_f)
                ll_last_scale_update = ll_current.copy()
                fully_converged = self.modelContainer.converged.copy()
            else:
                # Update intermediate convergence in self.model.converged.
                converged_f = np.logical_or(
                    ll_previous < ll_current,  # loss gets worse
                    np.abs(ll_previous - ll_current)
                    / np.maximum(  # relative decrease in loss is too small
                        np.nextafter(0, np.inf, dtype=ll_previous.dtype), np.abs(ll_previous)  # catch division by zero
                    )
                    < pkg_constants.LLTOL_BY_FEATURE,
                )
                self.modelContainer.converged = np.logical_or(self.modelContainer.converged, converged_f)
                if np.all(self.modelContainer.converged):
                    # All location models are converged. This means that the next update will be b model
                    # update and all remaining intermediate a model updates can be skipped:
                    epochs_until_scale_update = 0

            # Conclude and report iteration.
            train_step += 1
            # logging.getLogger("batchglm").info(
            sys.stdout.write(
                "iter %s: ll=%f, converged: %.2f%% (loc: %.2f%%, scale update: %s), in %.2fsec\n"
                % (
                    (" " if train_step < 10 else "") + (" " if train_step < 100 else "") + str(train_step),
                    np.sum(ll_current),
                    np.mean(fully_converged) * 100,
                    np.mean(self.modelContainer.converged) * 100,
                    str(epochs_until_scale_update == update_scale_freq),
                    time.time() - t0,
                )
            )
            # sys.stdout.write(
            #    '\riter %i: ll=%f, %.2f%% converged' %
            #    (train_step, np.sum(ll_current), np.round(np.mean(delayed_converged)*100, 2))
            # )
            # sys.stdout.flush()
            self.lls.append(ll_current)
        # sys.stdout.write('\r')
        # sys.stdout.flush()

    def iwls_step(self, idx_update: np.ndarray) -> np.ndarray:
        """
        A single step in IWLS
        :return: np.ndarray  (inferred param x features)
        """
        w = self.modelContainer.fim_weight_location_location_j(j=idx_update)  # (observations x features)
        ybar = self.modelContainer.ybar_j(j=idx_update)  # (observations x features)
        # Translate to problem of form ax = b for each feature:
        # (in the following, X=design and Y=counts)
        # a=X^T*W*X: ([features] x inferred param)
        # x=theta: ([features] x inferred param)
        # b=X^T*W*Ybar: ([features] x inferred param)
        xh = np.matmul(self.modelContainer.design_loc, self.modelContainer.constraints_loc)
        xhw = np.einsum("ob,of->fob", xh, w)
        a = np.einsum("fob,oc->fbc", xhw, xh)
        b = np.einsum("fob,of->fb", xhw, ybar)

        delta_theta = np.zeros_like(self.modelContainer.theta_location)

        if isinstance(a, dask.array.core.Array):
            # Have to use a workaround to solve problems in parallel in dask here. This workaround does
            # not work if there is only a single problem, ie. if the first dimension of a and b has length 1.
            if a.shape[0] != 1:
                invertible = np.where(
                    dask.array.map_blocks(
                        lambda x: np.expand_dims(np.expand_dims(np.linalg.cond(x, p=None), axis=-1), axis=-1),
                        a,
                        chunks=a.shape,
                    )
                    .squeeze()
                    .compute()
                    < 1 / sys.float_info.epsilon
                )[0]
                if len(idx_update[invertible]) > 1:
                    delta_theta[:, idx_update[invertible]] = (
                        dask.array.map_blocks(
                            np.linalg.solve, a[invertible], b[invertible, :, None], chunks=b[invertible, :, None].shape
                        )
                        .squeeze()
                        .T.compute()
                    )
                elif len(idx_update[invertible]) == 1:
                    delta_theta[:, idx_update[invertible]] = np.expand_dims(
                        np.linalg.solve(a[invertible[0]], b[invertible[0]]).compute(), axis=-1
                    )
            else:
                if np.linalg.cond(a.compute(), p=None) < 1 / sys.float_info.epsilon:
                    delta_theta[:, idx_update] = np.expand_dims(np.linalg.solve(a[0], b[0]).compute(), axis=-1)
                    invertible = np.array([0])
                else:
                    invertible = np.array([])
        else:
            invertible = np.where(np.linalg.cond(a, p=None) < 1 / sys.float_info.epsilon)[0]
            delta_theta[:, idx_update[invertible]] = np.linalg.solve(a[invertible], b[invertible]).T
        if invertible.shape[0] < len(idx_update):
            sys.stdout.write("caught %i linalg singular matrix errors\n" % (len(idx_update) - invertible.shape[0]))
        # Via np.linalg.lsts:
        # delta_theta[:, idx_update] = np.concatenate([
        #    np.expand_dims(np.linalg.lstsq(a[i, :, :], b[i, :])[0], axis=-1)
        #    for i in idx_update)
        # ], axis=-1)
        # Via np.linalg.inv:
        # #delta_theta[:, idx_update] = np.concatenate([
        #    np.expand_dims(np.matmul(np.linalg.inv(a[i, :, :]), b[i, :]), axis=-1)
        #    for i in idx_update)
        # ], axis=-1)
        return delta_theta

    def b_step(
        self, idx_update: np.ndarray, method: str, ftol: float, lr: float, max_iter: int, nproc: int
    ) -> np.ndarray:
        """
        A single step for the scale model
        :return:
        """
        if method.lower() in ["gd"]:
            return self._scale_step_gd(idx_update=idx_update, ftol=ftol, lr=lr, max_iter=max_iter)
        else:
            return self._scale_step_loop(
                idx_update=idx_update, method=method, ftol=ftol, max_iter=max_iter, nproc=nproc
            )

    def _scale_step_gd(self, idx_update: np.ndarray, ftol: float, max_iter: int, lr: float) -> np.ndarray:
        """
        A single gradient descent stop for the scale model
        :return:
        """
        iter = 0
        theta_scale_old = self.modelContainer.theta_scale.compute()
        converged = np.tile(True, self.modelContainer.num_features)
        converged[idx_update] = False
        ll_current = -self.modelContainer.ll_byfeature.compute()
        while np.any(np.logical_not(converged)) and iter < max_iter:
            idx_to_update = np.where(np.logical_not(converged))[0]
            jac = np.zeros_like(self.modelContainer.theta_scale).compute()
            # Use mean jacobian so that learning rate is independent of number of samples.
            jac[:, idx_to_update] = (
                self.modelContainer.jac_scale_j(j=idx_to_update).compute().T / self.modelContainer.num_observations
            )
            self.modelContainer.theta_scale_j_setter(
                value=(self.modelContainer.theta_scale.compute() + lr * jac)[:, idx_to_update], j=idx_to_update
            )
            # Assess convergence:
            ll_previous = ll_current
            ll_current = -self.modelContainer.ll_byfeature.compute()
            converged_f = (ll_current - ll_previous) / ll_previous > -ftol
            theta_scale_new = self.modelContainer.theta_scale.compute()
            theta_scale_new[:, converged_f] = theta_scale_new[:, converged_f] - lr * jac[:, converged_f]
            self.modelContainer.theta_scale = theta_scale_new
            converged = np.logical_or(converged, converged_f)
            iter += 1
            logging.getLogger("batchglm").info(f"iter {iter:>3}: ll={np.sum(ll_current)}, converged scale model: {np.mean(converged) * 100:.2f}")
        return self.modelContainer.theta_scale.compute() - theta_scale_old

    def optim_handle(self, b_j, data_j, eta_loc_j, xh_scale, max_iter, ftol):
        # Need to supply dense numpy array to scipy optimize:
        if isinstance(data_j, sparse.COO) or isinstance(data_j, scipy.sparse.csr_matrix):
            data_j = data_j.todense()
        if len(data_j.shape) == 1:
            data_j = np.expand_dims(data_j, axis=-1)

        ll = self.modelContainer.ll_handle()
        lb, ub = self.modelContainer.param_bounds(dtype=data_j.dtype)
        lb_bracket = np.max([lb["theta_scale"], b_j - 20])
        ub_bracket = np.min([ub["theta_scale"], b_j + 20])

        def cost_theta_scale(x, data_jj, eta_loc_jj, xh_scale_jj):
            x = np.clip(np.array([[x]]), lb["theta_scale"], ub["theta_scale"])
            return -np.sum(ll(data_jj, eta_loc_jj, x, xh_scale_jj))

        # jac_scale = self.model.jac_scale_handle()
        # def cost_theta_scale_prime(x, data_jj, eta_loc_jj, xh_scale_jj):
        #    x = np.clip(np.array([[x]]), lb["theta_scale"], ub["theta_scale"])
        #    return - np.sum(jac_scale(data_jj, eta_loc_jj, x, xh_scale_jj))
        # return scipy.optimize.line_search(
        #    f=cost_theta_scale,
        #    myfprime=cost_theta_scale_prime,
        #    args=(data_j, eta_loc_j, xh_scale),
        #    maxiter=max_iter,
        #    xk=b_j+5,
        #    pk=-np.ones_like(b_j)
        # )

        return scipy.optimize.brent(
            func=cost_theta_scale,
            args=(data_j, eta_loc_j, xh_scale),
            maxiter=max_iter,
            tol=ftol,
            brack=(lb_bracket, ub_bracket),
            full_output=True,
        )

    def _scale_step_loop(
        self, idx_update: np.ndarray, method: str, max_iter: int, ftol: float, nproc: int
    ) -> np.ndarray:
        """
        A single loop step for the scale model
        :return:
        """
        delta_theta = np.zeros(shape=self.modelContainer.theta_scale.shape)

        xh_scale = self.modelContainer.xh_scale.compute()
        theta_scale = self.modelContainer.theta_scale.compute()
        if nproc > 1 and len(idx_update) > nproc:
            sys.stdout.write(
                "\rFitting %i dispersion models: (progress not available with multiprocessing)" % len(idx_update)
            )
            sys.stdout.flush()
            with multiprocessing.Pool(processes=nproc) as pool:
                x = self.modelContainer.x.compute()
                eta_loc = self.modelContainer.eta_loc.compute()
                results = pool.starmap(
                    self.optim_handle,
                    [(theta_scale[0, j], x[:, [j]], eta_loc[:, [j]], xh_scale, max_iter, ftol) for j in idx_update],
                )
                pool.close()
            delta_theta[0, idx_update] = np.array([x[0] for x in results])
            sys.stdout.write("\r")
            sys.stdout.flush()
        else:
            t0 = time.time()
            for i, j in enumerate(idx_update):
                sys.stdout.write(
                    "\rFitting dispersion models: %.2f%% in %.2fsec"
                    % (np.round(i / len(idx_update) * 100.0, 2), time.time() - t0)
                )
                sys.stdout.flush()
                if method.lower() == "brent":
                    eta_loc = self.modelContainer.eta_loc_j(j=j).compute()
                    data = self.modelContainer.x_j(j=j).compute()
                    # Need to supply dense numpy array to scipy optimize:
                    if isinstance(data, sparse.COO) or isinstance(data, scipy.sparse.csr_matrix):
                        data = data.todense()

                    ll = self.modelContainer.ll_handle()
                    lb, ub = self.modelContainer.param_bounds(dtype=data.dtype)
                    lb_bracket = np.max([lb["theta_scale"], theta_scale[0, j] - 20])
                    ub_bracket = np.min([ub["theta_scale"], theta_scale[0, j] + 20])

                    def cost_theta_scale(x, data_j, eta_loc_j, xh_scale_j):
                        x = np.clip(np.array([[x]]), lb["theta_scale"], ub["theta_scale"])
                        return -np.sum(ll(data_j, eta_loc_j, x, xh_scale_j))

                    delta_theta[0, j] = scipy.optimize.brent(
                        func=cost_theta_scale,
                        args=(data, eta_loc, xh_scale),
                        maxiter=max_iter,
                        tol=ftol,
                        brack=(lb_bracket, ub_bracket),
                        full_output=False,
                    )
                else:
                    raise ValueError("method %s not recognized" % method)
            sys.stdout.write("\r")
            sys.stdout.flush()
        
        if isinstance(self.modelContainer.theta_scale, dask.array.core.Array):
            delta_theta[:, idx_update] -= self.modelContainer.theta_scale_j(j=idx_update).compute()
        else:
            delta_theta[:, idx_update] -= self.modelContainer.theta_scale_j(j=idx_update).copy()
        return delta_theta

    def finalize(self):
        """
        Evaluate all tensors that need to be exported from session and save these as class attributes
        and close session.

        Changes .model entry from tf1-based EstimatorGraph to numpy based Model instance and
        transfers relevant attributes.
        """
        # Read from numpy-IRLS estimator specific model:
        self.modelContainer._hessian = -self.modelContainer.fim.compute()
        fisher_inv = np.zeros_like(self.modelContainer._hessian)
        invertible = np.where(np.linalg.cond(self.modelContainer._hessian, p=None) < 1 / sys.float_info.epsilon)[0]
        fisher_inv[invertible] = np.linalg.inv(-self.modelContainer._hessian[invertible])
        self.modelContainer._fisher_inv = fisher_inv
        self.modelContainer._jacobian = np.sum(np.abs(self.modelContainer.jac.compute() / self.modelContainer.x.shape[0]), axis=1)
        self.modelContainer._log_likelihood = self.modelContainer.ll_byfeature.compute()
        self.modelContainer._loss = np.sum(self.modelContainer._log_likelihood)

    def get_model_container(self, input_data):
        """Deprecated: This is equivalent to self.modelContainer now and can be removed"""
        # return Model(input_data=input_data)
        return self.modelContainer
