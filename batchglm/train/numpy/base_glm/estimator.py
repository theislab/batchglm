import abc
import dask.array
import logging
import multiprocessing
import numpy as np
import pprint
import scipy
import scipy.optimize
import sys

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

    def train_sequence(
            self,
            training_strategy: str = "DEFAULT"
    ):
        if isinstance(training_strategy, str):
            training_strategy = self.TrainingStrategies[training_strategy].value[0]

        if training_strategy is None:
            training_strategy = self.TrainingStrategies.DEFAULT.value

        logging.getLogger("batchglm").info("training strategy:\n%s", pprint.pformat(training_strategy))
        self.train(**training_strategy)

    def train(
            self,
            max_steps: int,
            method_b: str = "brent",
            update_b_freq: int = 5,
            ftol_b: float = 1e-8,
            lr_b: float = 1e-2,
            max_iter_b: int = 100
    ):
        # Iterate until conditions are fulfilled.
        train_step = 0
        delayed_converged = np.tile(False, self.model.model_vars.n_features)

        ll_current = - self.model.ll_byfeature.compute()
        logging.getLogger("batchglm").info("iter %i: ll=%f" % (0, np.sum(ll_current)))
        while np.any(np.logical_not(delayed_converged)) and \
                train_step < max_steps:
            # Update parameters:
            # Line search step for scale model:
            if train_step % update_b_freq == 0 and train_step > 0:
                if isinstance(self.model.b_var, dask.array.core.Array):
                    b_var_cache = self.model.b_var.compute()
                else:
                    b_var_cache = self.model.b_var.copy()
                self.model.b_var = self.b_step(
                    idx=np.where(np.logical_not(delayed_converged))[0],
                    method=method_b,
                    ftol=ftol_b,
                    lr=lr_b,
                    max_iter=max_iter_b
                )
                # Reverse update by feature if update leads to worse loss:
                ll_proposal = - self.model.ll_byfeature.compute()
                if isinstance(self.model.b_var, dask.array.core.Array):
                    b_var_new = self.model.b_var.compute()
                else:
                    b_var_new = self.model.b_var.copy()
                b_var_new[:, ll_proposal > ll_current] = b_var_cache[:, ll_proposal > ll_current]
                self.model.b_var = b_var_new
                delayed_converged = self.model.converged.copy()
            # IWLS step for location model:
            self.model.a_var = self.model.a_var + self.iwls_step()

            # Evaluate convergence
            ll_previous = ll_current
            ll_current = - self.model.ll_byfeature.compute()
            converged_f = (ll_previous - ll_current) / ll_previous < pkg_constants.LLTOL_BY_FEATURE
            # Location model convergence status has to be updated if b model was updated
            if train_step % update_b_freq == 0 and train_step > 0:
                self.model.converged = converged_f
                delayed_converged = converged_f
            else:
                self.model.converged = np.logical_or(self.model.converged, converged_f)
            train_step += 1
            logging.getLogger("batchglm").info(
                "iter %i: ll=%f, converged location model: %.2f%%, converged total: %.2f%%" %
                (train_step, np.sum(ll_current), np.mean(self.model.converged)*100, np.mean(delayed_converged)*100)
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
                (iter, np.sum(ll_current), np.mean(converged) * 100)
            )
        return self.model.a_var.compute()

    def iwls_step(self) -> np.ndarray:
        """

        :return: (inferred param x features)
        """
        w = self.model.fim_weight_j(j=self.model.idx_not_converged)  # (observations x features)
        ybar = self.model.ybar_j(j=self.model.idx_not_converged)  # (observations x features)
        # Translate to problem of form ax = b for each feature:
        # (in the following, X=design and Y=counts)
        # a=X^T*W*X: ([features] x inferred param)
        # x=theta: ([features] x inferred param)
        # b=X^T*W*Ybar: ([features] x inferred param)
        xh = np.matmul(self.model.design_loc, self.model.constraints_loc)
        xhw = np.einsum('ob,of->fob', xh, w)
        a = np.einsum('fob,oc->fbc', xhw, xh)
        b = np.einsum('fob,of->fb', xhw, ybar)
        # Via np.linalg.solve:
        delta_theta = np.zeros_like(self.model.a_var)
        if isinstance(delta_theta, dask.array.core.Array):
            delta_theta = delta_theta.compute()

        if isinstance(a, dask.array.core.Array):
            # Have to use a workaround to solve problems in parallel in dask here. This workaround does
            # not work if there is only a single problem, ie. if the first dimension of a and b has length 1.
            if a.shape[0] != 1:
                delta_theta[:, self.model.idx_not_converged] = dask.array.map_blocks(
                    np.linalg.solve, a, b[:, :, None], chunks=b[:, :, None].shape
                ).squeeze().T.compute()
            else:
                delta_theta[:, self.model.idx_not_converged] = np.expand_dims(
                    np.linalg.solve(a[0], b[0]).compute(),
                    axis=-1
                )
        else:
            delta_theta[:, self.model.idx_not_converged] = np.linalg.solve(a, b).T
        # Via np.linalg.lsts:
        #delta_theta[:, self.idx_not_converged] = np.concatenate([
        #    np.expand_dims(np.linalg.lstsq(a[i, :, :], b[i, :])[0], axis=-1)
        #    for i in self.idx_not_converged)
        #], axis=-1)
        # Via np.linalg.inv:
        # #delta_theta[:, self.idx_not_converged] = np.concatenate([
        #    np.expand_dims(np.matmul(np.linalg.inv(a[i, :, :]), b[i, :]), axis=-1)
        #    for i in self.idx_not_converged)
        #], axis=-1)
        return delta_theta

    def b_step(
            self,
            idx: np.ndarray,
            method: str,
            ftol: float,
            lr: float,
            max_iter: int
    ) -> np.ndarray:
        """

        :return:
        """
        if method.lower() in ["gd"]:
            return self._b_step_gd(idx=idx, ftol=ftol, lr=lr, max_iter=max_iter)
        else:
            return self._b_step_loop(idx=idx, method=method, ftol=ftol, max_iter=max_iter)

    def _b_step_gd(
            self,
            idx: np.ndarray,
            ftol: float,
            max_iter: int,
            lr: float
    ) -> np.ndarray:
        """

        :return:
        """
        iter = 0
        converged = np.tile(True, self.model.model_vars.n_features)
        converged[idx] = False
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
                (iter, np.sum(ll_current), np.mean(converged) * 100)
            )
        return self.model.b_var.compute()

    def _b_step_loop(
            self,
            idx: np.ndarray,
            method: str,
            max_iter: int,
            ftol: float
    ) -> np.ndarray:
        """

        :return:
        """
        x0 = -10

        def cost_b_var(x):
            self.model.b_var_j_setter(value=x, j=j)
            return - np.sum(self.model.ll_j(j=j)).compute()

        def grad_b_var(x):
            self.model.b_var_j_setter(value=x, j=j)
            return - self.model.jac_b_j(j=j).compute()

        if isinstance(self.model.b_var, dask.array.core.Array):
            b_var_new = self.model.b_var.compute()
        else:
            b_var_new = self.model.b_var.copy()
        if False:
            # TODO support multithreading
            with multiprocessing.Pool(processes=3) as pool:
                if method.lower() == "brent":
                    def optim_handle(j):
                        scipy.optimize.brent(
                            func=cost_b_var,
                            args=(),
                            tol=ftol,
                            full_output=True,
                            maxiter=max_iter
                        )
                else:
                    raise ValueError("method %s not recognized" % method)
                results = pool.starmap(optim_handle, idx)
                b_var_new[0, :] = np.array([x[0] for x in results])
        else:
            for i, j in enumerate(idx):
                sys.stdout.write('\rFitting dispersion models in progress: %.2f%%' % np.round(i/len(idx)*100., 2))
                sys.stdout.flush()
                if method.lower() == "linesearch":
                    ls_result = scipy.optimize.line_search(
                        f=cost_b_var,
                        myfprime=grad_b_var,
                        xk=np.array([x0]),
                        pk=np.array([1.]),
                        gfk=None,
                        old_fval=None,
                        old_old_fval=None,
                        args=(),
                        c1=0.0001,
                        c2=0.9,
                        amax=50.,
                        extra_condition=None,
                        maxiter=max_iter
                    )
                    b_var_new[0, j] = x0 + ls_result[0]
                elif method.lower() == "brent":
                    b_var_new[0, j] = scipy.optimize.brent(
                        func=cost_b_var,
                        args=(),
                        maxiter=max_iter,
                        tol=ftol,
                        brack=(-5, 5),
                        full_output=False
                    )
                else:
                    raise ValueError("method %s not recognized" % method)
            sys.stdout.write('\r')
            sys.stdout.flush()
        return b_var_new

    def finalize(self):
        """
        Evaluate all tensors that need to be exported from session and save these as class attributes
        and close session.

        Changes .model entry from tf1-based EstimatorGraph to numpy based Model instance and
        transfers relevant attributes.
        """
        # Read from numpy-IRLS estimator specific model:

        self._hessian = self.model.hessian.compute()
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

