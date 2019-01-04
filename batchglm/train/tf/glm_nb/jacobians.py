import logging

import tensorflow as tf

from .model import BasicModelGraph, ModelVars
from .external import JacobiansGLM, JacobiansTF
from .external import op_utils
from .external import pkg_constants

logger = logging.getLogger(__name__)


class JacobiansAnalytic:
    """
    Compute the analytic Jacobian matrix for a negative binomial GLM.
    """

    def _coef_invariant_a(
            self,
            X,
            mu,
            r,
    ):
        """
        Compute the coefficient index invariant part of the
        mean model gradient of base_glm_all model.

        Below, X are design matrices of the mean (m)
        and dispersion (r) model respectively, Y are the
        observed data. Const is constant across all combinations
        of i and j.
        .. math::

            &J^{m}_{i} = X^m_i*\bigg(Y-(Y+r)*\frac{mu}{mu+r}\bigg) \\
            &const = Y-(Y+r)*\frac{mu}{mu+r} \\
            &J^{m}_{i} = X^m_i*const \\

        :param X: tf.tensor observations x features
            Observation by observation and feature.
        :param mu: tf.tensor observations x features
            Value of mean model by observation and feature.
        :param r: tf.tensor observations x features
            Value of dispersion model by observation and feature.

        :return const: tf.tensor observations x features
            Coefficient invariant terms of hessian of
            given observations and features.
        """
        const = tf.multiply(  # [observations, features]
            tf.add(X, r),
            tf.divide(
                mu,
                tf.add(mu, r)
            )
        )
        const = tf.subtract(X, const)
        return const


    def _coef_invariant_b(
            self,
            X,
            mu,
            r,
    ):
        """
        Compute the coefficient index invariant part of the
        dispersion model gradient of base_glm_all model.

        Below, X are design matrices of the mean (m)
        and dispersion (r) model respectively, Y are the
        observed data. Const is constant across all combinations
        of i and j.
        .. math::

            J{r}_{i} &= X^r_i \\
                &*r*\bigg(psi_0(r+Y)-psi_0(r) \\
                &-\frac{r+Y}{r+mu} \\
                &+log(r)+1-log(r+mu) \bigg) \\
            const = r*\bigg(psi_0(r+Y)-psi_0(r) \\ const1
                &-\frac{r+Y}{r+mu} \\ const2
                &+log(r)+1-log(r+mu) \bigg) \\ const3
            J^{r}_{i} &= X^r_i * const \\

        :param X: tf.tensor observations x features
            Observation by observation and feature.
        :param mu: tf.tensor observations x features
            Value of mean model by observation and feature.
        :param r: tf.tensor observations x features
            Value of dispersion model by observation and feature.

        :return const: tf.tensor observations x features
            Coefficient invariant terms of hessian of
            given observations and features.
        """
        scalar_one = tf.constant(1, shape=(), dtype=X.dtype)
        # Pre-define sub-graphs that are used multiple times:
        r_plus_mu = r + mu
        r_plus_x = r + X
        # Define graphs for individual terms of constant term of hessian:
        const1 = tf.subtract(
            tf.math.digamma(x=r_plus_x),
            tf.math.digamma(x=r)
        )
        const2 = tf.negative(r_plus_x / r_plus_mu)
        const3 = tf.add(
            tf.log(r),
            scalar_one - tf.log(r_plus_mu)
        )
        const = tf.add_n([const1, const2, const3])  # [observations, features]
        const = r * const
        return const

    def analytic(
            self,
            batched_data,
            sample_indices,
            constraints_loc,
            constraints_scale,
            model_vars: ModelVars,
            iterator,
            dtype
    ):
        """
        Compute the closed-form of the base_glm_all model jacobian
        by evalutating its terms grouped by observations.
        """

        def _a_byobs(X, design_loc, constraints_loc, mu, r):
            """
            Compute the mean model block of the jacobian.

            :param X: tf.tensor observations x features
                Observation by observation and feature.
            :param mu: tf.tensor observations x features
                Value of mean model by observation and feature.
            :param r: tf.tensor observations x features
                Value of dispersion model by observation and feature.
            :return Jblock: tf.tensor features x coefficients
                Block of jacobian.
            """
            W = self._coef_invariant_a(X=X, mu=mu, r=r)  # [observations, features]
            XH = tf.matmul(design_loc, constraints_loc)
            Jblock = tf.matmul(tf.transpose(W), XH)  # [features, coefficients]
            return Jblock

        def _b_byobs(X, design_scale, constraints_scale, mu, r):
            """
            Compute the dispersion model block of the jacobian.
            """
            W = self._coef_invariant_b(X=X, mu=mu, r=r)  # [observations, features]
            XH = tf.matmul(design_scale, constraints_scale)
            Jblock = tf.matmul(tf.transpose(W), XH)  # [features, coefficients]
            return Jblock

        def _assemble_bybatch(idx, data):
            """
            Assemble jacobian of a batch of observations across all features.

            This function runs the data batch (an observation) through the
            model graph and calls the wrappers that compute the
            individual closed forms of the jacobian.

            :param data: tuple
                Containing the following parameters:
                - X: tf.tensor observations x features
                    Observation by observation and feature.
                - size_factors: tf.tensor observations x features
                    Model size factors by observation and feature.
                - params: tf.tensor features x coefficients
                    Estimated model variables.
            :return J: tf.tensor features x coefficients
                Jacobian evaluated on a single observation, provided in data.
            """
            X, design_loc, design_scale, size_factors = data

            model = BasicModelGraph(
                X=X,
                design_loc=design_loc,
                design_scale=design_scale,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                a=model_vars.a,
                b=model_vars.b,
                dtype=dtype,
                size_factors=size_factors
            )
            mu = model.mu
            r = model.r

            if self._compute_jac_a and self._compute_jac_b:
                J_a = _a_byobs(X=X, design_loc=design_loc, constraints_loc=constraints_loc, mu=mu, r=r)
                J_b = _b_byobs(X=X, design_scale=design_scale, constraints_scale=constraints_scale, mu=mu, r=r)
                J = tf.concat([J_a, J_b], axis=1)
            elif self._compute_jac_a and not self._compute_jac_b:
                J = _a_byobs(X=X, design_loc=design_loc, constraints_loc=constraints_loc, mu=mu, r=r)
            elif not self._compute_jac_a and self._compute_jac_b:
                J = _b_byobs(X=X, design_scale=design_scale, constraints_scale=constraints_scale, mu=mu, r=r)
            else:
                raise ValueError("either require jac_a or jac_b")

            return J

        def _red(prev, cur):
            """
            Reduction operation for jacobian computation across observations.

            Every evaluation of the jacobian on an observation yields a full
            jacobian matrix. This function sums over consecutive evaluations
            of this hessian so that not all seperate evluations have to be
            stored.
            """
            return tf.add(prev, cur)

        if iterator:
            J = op_utils.map_reduce(
                last_elem=tf.gather(sample_indices, tf.size(sample_indices) - 1),
                data=batched_data,
                map_fn=_assemble_bybatch,
                reduce_fn=_red,
                parallel_iterations=pkg_constants.TF_LOOP_PARALLEL_ITERATIONS
            )
        else:
            J = _assemble_bybatch(
                idx=sample_indices,
                data=batched_data
            )

        return J


class Jacobians(JacobiansAnalytic, JacobiansTF, JacobiansGLM):
    """
    Jacobian matrix computation interface for negative binomial GLMs.
    """