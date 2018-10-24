from typing import List

import tensorflow as tf
# import numpy as np

import logging

from .base import BasicModelGraph, ModelVars

from .external import op_utils
from .external import pkg_constants

logger = logging.getLogger(__name__)


def _coef_invariant_ab(
        X,
        mu,
        r,
):
    """
    Compute the coefficient index invariant part of the
    mean-dispersion model block of the hessian of nb_glm model.

    Note that there are two blocks of the same size which can
    be compute from each other with a transpose operation as
    the hessian is symmetric.

    Below, X are design matrices of the mean (m)
    and dispersion (r) model respectively, Y are the
    observed data. Const is constant across all combinations
    of i and j.
    .. math::

        &H^{m,r}_{i,j} = X^m_i*X^r_j*mu*\frac{Y-mu}{(1+mu/r)^2} \\
        &H^{r,m}_{i,j} = X^m_i*X^r_j*r*mu*\frac{Y-mu}{(mu+r)^2} \\
        &const = r*mu*\frac{Y-mu}{(mu+r)^2} \\
        &H^{m,r}_{i,j} = X^m_i*X^r_j*const \\
        &H^{r,m}_{i,j} = X^m_i*X^r_j*const \\

    :param X: tf.tensor observations x features
        Observation by observation and feature.
    :param mu: tf.tensor observations x features
        Value of mean model by observation and feature.
    :param r: tf.tensor observations x features
        Value of dispersion model by observation and feature.
    :param dtype: dtype
    :return const: tf.tensor observations x features
        Coefficient invariant terms of hessian of
        given observations and features.
    """
    const = tf.multiply(
        mu * r,  # [observations, features]
        tf.divide(
            X - mu,  # [observations, features]
            tf.square(mu + r)
        )
    )
    return const


def _coef_invariant_aa(
        X,
        mu,
        r,
):
    """
    Compute the coefficient index invariant part of the
    mean model block of the hessian of nb_glm model.

    Below, X are design matrices of the mean (m)
    and dispersion (r) model respectively, Y are the
    observed data. Const is constant across all combinations
    of i and j.
    .. math::

        &H^{m,m}_{i,j} = -X^m_i*X^m_j*mu*\frac{Y/r+1}{(1+mu/r)^2} \\
        &const = -mu*\frac{Y/r+1}{(1+mu/r)^2} \\
        &H^{m,m}_{i,j} = X^m_i*X^m_j*const \\

    :param X: tf.tensor observations x features
        Observation by observation and feature.
    :param mu: tf.tensor observations x features
        Value of mean model by observation and feature.
    :param r: tf.tensor observations x features
        Value of dispersion model by observation and feature.
    :param dtype: dtype
    :return const: tf.tensor observations x features
        Coefficient invariant terms of hessian of
        given observations and features.
    """
    const = tf.negative(tf.multiply(
        mu,  # [observations, features]
        tf.divide(
            (X / r) + 1,
            tf.square((mu / r) + 1)
        )
    ))
    return const


def _coef_invariant_bb(
        X,
        mu,
        r,
):
    """
    Compute the coefficient index invariant part of the
    dispersion model block of the hessian of nb_glm model.

    Below, X are design matrices of the mean (m)
    and dispersion (r) model respectively, Y are the
    observed data. Const is constant across all combinations
    of i and j.
    .. math::

        H^{r,r}_{i,j}&= X^r_i*X^r_j \\
            &*r*\bigg(psi_0(r+Y)+r*psi_1(r+Y) \\
            &+psi_0(r)+r*psi_1(r) \\
            &-\frac{mu*(r+X)+2*r*(r+m)}{(r+mu)^2} \\
            &+log(r)+1-log(r+mu) \bigg) \\
        const = r*\bigg(psi_0(r+Y)+r*psi_1(r+Y) \\ const1
            &+psi_0(r)+r*psi_1(r) \\ const2
            &-\frac{mu*(r+X)+2*r*(r+m)}{(r+mu)^2} \\ const3
            &+log(r)+1-log(r+mu) \bigg) \\ const4
        H^{r,r}_{i,j}&= X^r_i*X^r_j * const \\

    :param X: tf.tensor observations x features
        Observation by observation and feature.
    :param mu: tf.tensor observations x features
        Value of mean model by observation and feature.
    :param r: tf.tensor observations x features
        Value of dispersion model by observation and feature.
    :param dtype: dtype
    :return const: tf.tensor observations x features
        Coefficient invariant terms of hessian of
        given observations and features.
    """
    scalar_one = tf.constant(1, shape=(), dtype=X.dtype)
    scalar_two = tf.constant(2, shape=(), dtype=X.dtype)
    # Pre-define sub-graphs that are used multiple times:
    r_plus_mu = r + mu
    r_plus_x = r + X
    # Define graphs for individual terms of constant term of hessian:
    const1 = tf.add(  # [observations, features]
        tf.math.digamma(x=r_plus_x),
        r * tf.math.polygamma(a=scalar_one, x=r_plus_x)
    )
    const2 = tf.negative(tf.add(  # [observations, features]
        tf.math.digamma(x=r),
        r * tf.math.polygamma(a=scalar_one, x=r)
    ))
    const3 = tf.negative(tf.divide(
        tf.add(
            mu * r_plus_x,
            scalar_two * r * r_plus_mu
        ),
        tf.square(r_plus_mu)
    ))
    const4 = tf.add(  # [observations, features]
        tf.log(r),
        scalar_two - tf.log(r_plus_mu)
    )
    const = tf.add_n([const1, const2, const3, const4])  # [observations, features]
    const = tf.multiply(r, const)
    return const


class Hessians:
    """ Compute the nb_glm model hessian.
    """
    hessian: tf.Tensor
    neg_hessian: tf.Tensor

    def __init__(
            self,
            batched_data: tf.data.Dataset,
            singleobs_data: tf.data.Dataset,
            sample_indices: tf.Tensor,
            constraints_loc,
            constraints_scale,
            model_vars: ModelVars,
            dtype,
            mode="obs",
            iterator=True
    ):
        """ Return computational graph for hessian based on mode choice.

        :param batched_data:
            Dataset iterator over mini-batches of data (used for training) or tf.Tensors of mini-batch.
        :param singleobs_data: Dataset iterator over single observation batches of data.
        :param sample_indices: Indices of samples to be used.
        :param constraints_loc: Constraints for location model.
            Array with constraints in rows and model parameters in columns.
            Each constraint contains non-zero entries for the a of parameters that
            has to sum to zero. This constraint is enforced by binding one parameter
            to the negative sum of the other parameters, effectively representing that
            parameter as a function of the other parameters. This dependent
            parameter is indicated by a -1 in this array, the independent parameters
            of that constraint (which may be dependent at an earlier constraint)
            are indicated by a 1.
        :param constraints_scale: Constraints for scale model.
            Array with constraints in rows and model parameters in columns.
            Each constraint contains non-zero entries for the a of parameters that
            has to sum to zero. This constraint is enforced by binding one parameter
            to the negative sum of the other parameters, effectively representing that
            parameter as a function of the other parameters. This dependent
            parameter is indicated by a -1 in this array, the independent parameters
            of that constraint (which may be dependent at an earlier constraint)
            are indicated by a 1.
        :param mode: str
            Mode by with which hessian is to be evaluated,
            for analytic solutions of the hessian one can either chose by
            "feature" or by "obs" (observation). Note that sparse
            observation matrices X are often csr, ie. slicing is
            faster by row/observation, so that hessian evaluation
            by observation is much faster. "tf" allows for
            evaluation of the hessian via the tf.hessian function,
            which is done by feature for implementation reasons.
        :param iterator: bool
            Whether an iterator or a tensor (single yield of an iterator) is given
            in
        """
        if constraints_loc != None and mode != "tf":
            raise ValueError("closed form hessian does not work if constraints_loc is not None")
        if constraints_scale != None and mode != "tf":
            raise ValueError("closed form hessian does not work if constraints_scale is not None")

        if mode == "obs":
            logger.info("Performance warning for hessian mode: " +
                        "obs_batched is strongly recommended as an alternative to obs.")
            self.hessian = self.byobs(
                batched_data=singleobs_data,
                sample_indices=sample_indices,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                model_vars=model_vars,
                batched=False,
                iterator=iterator,
                dtype=dtype
            )
            self.neg_hessian = tf.negative(self.hessian)
        elif mode == "obs_batched":
            self.hessian = self.byobs(
                batched_data=batched_data,
                sample_indices=sample_indices,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                model_vars=model_vars,
                batched=True,
                iterator=iterator,
                dtype=dtype
            )
            self.neg_hessian = tf.negative(self.hessian)
        elif mode == "feature":
            self.hessian = self.byfeature(
                batched_data=batched_data,
                sample_indices=sample_indices,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                model_vars=model_vars,
                dtype=dtype
            )
            self.neg_hessian = tf.negative(self.hessian)
        elif mode == "tf":
            # tensorflow computes the hessian based on the objective,
            # which is the negative loglikelihood. Accordingly, the hessian
            # is the negative hessian.
            self.neg_hessian = self.tf_byfeature(
                batched_data=batched_data,
                sample_indices=sample_indices,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                model_vars=model_vars,
                dtype=dtype
            )
            self.hessian = tf.negative(self.neg_hessian)
        else:
            raise ValueError("mode not recognized in hessian_nb_glm: " + mode)

    def byobs(
            self,
            batched_data,
            sample_indices,
            constraints_loc,
            constraints_scale,
            model_vars: ModelVars,
            batched,
            iterator,
            dtype
    ):
        """
        Compute the closed-form of the nb_glm model hessian
        by evaluating its terms grouped by observations.

        Has three sub-functions which built the specific blocks of the hessian
        and one sub-function which concatenates the blocks into a full hessian.

        Note that two different groups of functions compute the hessian
        block either with standard matrix multiplication for a single observation
        at a time which expect the output of an iterator which only yields one
        observation at a time. The behaviour of these functions is save in terms
        of memory usage. There is a second set of functions (*batched) which
        use the einsum to compute the hessian block on a batch of observations
        in a single go. This requires the handling of a latent 4D tensor which
        potentially large memory usage, depending on the einsum behaviour. In
        principle the latter can be fast though as they replace iterations which
        larger tensor operations.
        """

        def _aa_byobs(X, design_loc, design_scale, mu, r):
            """
            Compute the mean model diagonal block of the
            closed form hessian of nb_glm model by observation across features.

            :param X: tf.tensor observations x features
                Observation by observation and feature.
            :param mu: tf.tensor observations x features
                Value of mean model by observation and feature.
            :param r: tf.tensor observations x features
                Value of dispersion model by observation and feature.
            """
            const = _coef_invariant_aa(  # [observations=1 x features]
                X=X,
                mu=mu,
                r=r,
            )
            nonconst = tf.matmul(tf.transpose(design_loc), design_loc)  # [coefficients, coefficients]
            nonconst = tf.expand_dims(nonconst, axis=0)  # [observations=1, coefficients, coefficients]
            Hblock = tf.tensordot(  # [features, coefficients, coefficients]
                a=tf.transpose(const),  # [features, observations=1]
                b=nonconst,  # [observations=1, coefficients, coefficients]
                axes=1  # collapse last dimension of a and first dimension of b
            )
            return Hblock

        def _bb_byobs(X, design_loc, design_scale, mu, r):
            """
            Compute the dispersion model diagonal block of the
            closed form hessian of nb_glm model by observation across features.
            """
            const = _coef_invariant_bb(  # [observations=1 x features]
                X=X,
                mu=mu,
                r=r,
            )
            nonconst = tf.matmul(tf.transpose(design_scale), design_scale)  # [coefficients, coefficients]
            nonconst = tf.expand_dims(nonconst, axis=0)  # [observations=1, coefficients, coefficients]
            Hblock = tf.tensordot(  # [features, coefficients, coefficients]
                a=tf.transpose(const),  # [features, observations=1]
                b=nonconst,  # [observations=1, coefficients, coefficients]
                axes=1  # collapse last dimension of a and first dimension of b
            )
            return Hblock

        def _ab_byobs(X, design_loc, design_scale, mu, r):
            """
            Compute the mean-dispersion model off-diagonal block of the
            closed form hessian of nb_glm model by observation across features.

            Note that there are two blocks of the same size which can
            be compute from each other with a transpose operation as
            the hessian is symmetric.
            """
            const = _coef_invariant_ab(  # [observations=1 x features]
                X=X,
                mu=mu,
                r=r,
            )
            nonconst = tf.matmul(tf.transpose(design_loc), design_scale)  # [coefficient_loc, coefficients_scale]
            nonconst = tf.expand_dims(nonconst, axis=0)  # [observations=1, coefficient_loc, coefficients_scale]
            Hblock = tf.tensordot(  # [features, coefficient_loc, coefficients_scale]
                a=tf.transpose(const),  # [features, observations=1]
                b=nonconst,  # [observations=1, coefficient_loc, coefficients_scale]
                axes=1  # collapse last dimension of a and first dimension of b
            )
            return Hblock

        def _aa_byobs_batched(X, design_loc, design_scale, mu, r):
            """
            Compute the mean model diagonal block of the
            closed form hessian of nb_glm model by observation across features
            for a batch of observations.

            :param X: tf.tensor observations x features
                Observation by observation and feature.
            :param mu: tf.tensor observations x features
                Value of mean model by observation and feature.
            :param r: tf.tensor observations x features
                Value of dispersion model by observation and feature.
            """
            scalar_one = tf.constant(1, shape=[1, 1], dtype=dtype)
            const = _coef_invariant_aa(  # [observations x features]
                X=X,
                mu=mu,
                r=r,
            )
            # The computation of the hessian block requires two outer products between
            # feature-wise constants and the coefficient wise design matrix entries, for each observation.
            # The resulting tensor is observations x features x coefficients x coefficients which
            # is too large too store in memory in most cases. However, the full 4D tensor is never
            # actually needed but only its marginal across features, the final hessian block shape.
            # Here, we use the einsum to efficiently perform the two outer products and the marginalisation.
            Hblock = tf.einsum('ofc,od->fcd',
                               tf.einsum('of,oc->ofc', const, design_loc),
                               design_loc)
            return Hblock

        def _bb_byobs_batched(X, design_loc, design_scale, mu, r):
            """
            Compute the dispersion model diagonal block of the
            closed form hessian of nb_glm model by observation across features.
            """
            const = _coef_invariant_bb(  # [observations=1 x features]
                X=X,
                mu=mu,
                r=r,
            )
            # The computation of the hessian block requires two outer products between
            # feature-wise constants and the coefficient wise design matrix entries, for each observation.
            # The resulting tensor is observations x features x coefficients x coefficients which
            # is too large too store in memory in most cases. However, the full 4D tensor is never
            # actually needed but only its marginal across features, the final hessian block shape.
            # Here, we use the Einstein summation to efficiently perform the two outer products and the marginalisation.
            Hblock = tf.einsum('ofc,od->fcd',
                               tf.einsum('of,oc->ofc', const, design_scale),
                               design_scale)
            return Hblock

        def _ab_byobs_batched(X, design_loc, design_scale, mu, r):
            """
            Compute the mean-dispersion model off-diagonal block of the
            closed form hessian of nb_glm model by observastion across features.

            Note that there are two blocks of the same size which can
            be compute from each other with a transpose operation as
            the hessian is symmetric.
            """
            const = _coef_invariant_ab(  # [observations=1 x features]
                X=X,
                mu=mu,
                r=r,
            )
            # The computation of the hessian block requires two outer products between
            # feature-wise constants and the coefficient wise design matrix entries, for each observation.
            # The resulting tensor is observations x features x coefficients x coefficients which
            # is too large too store in memory in most cases. However, the full 4D tensor is never
            # actually needed but only its marginal across features, the final hessian block shape.
            # Here, we use the Einstein summation to efficiently perform the two outer products and the marginalisation.
            Hblock = tf.einsum('ofc,od->fcd',
                               tf.einsum('of,oc->ofc', const, design_loc),
                               design_scale)
            return Hblock

        def _assemble_byobs(idx, data):
            """
            Assemble hessian of a single observation across all features.

            This function runs the data batch (an observation) through the
            model graph and calls the wrappers that compute the
            individual closed forms of the hessian.

            :param data: tuple
                Containing the following parameters:
                - X: tf.tensor observations x features
                    Observation by observation and feature.
                - size_factors: tf.tensor observations x features
                    Model size factors by observation and feature.
                - params: tf.tensor features x coefficients
                    Estimated model variables.
            :return H: tf.tensor features x coefficients x coefficients
                Hessian evaluated on a single observation, provided in data.
            """
            X, design_loc, design_scale, size_factors = data
            a_split, b_split = tf.split(params, tf.TensorShape([p_shape_a, p_shape_b]))

            model = BasicModelGraph(
                X=X,
                design_loc=design_loc,
                design_scale=design_scale,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                a=a_split,
                b=b_split,
                dtype=dtype,
                size_factors=size_factors
            )
            mu = model.mu
            r = model.r

            if batched:
                H_aa = _aa_byobs_batched(X=X, design_loc=design_loc, design_scale=design_scale, mu=mu, r=r)
                H_bb = _bb_byobs_batched(X=X, design_loc=design_loc, design_scale=design_scale, mu=mu, r=r)
                H_ab = _ab_byobs_batched(X=X, design_loc=design_loc, design_scale=design_scale, mu=mu, r=r)
            else:
                H_aa = _aa_byobs(X=X, design_loc=design_loc, design_scale=design_scale, mu=mu, r=r)
                H_bb = _bb_byobs(X=X, design_loc=design_loc, design_scale=design_scale, mu=mu, r=r)
                H_ab = _ab_byobs(X=X, design_loc=design_loc, design_scale=design_scale, mu=mu, r=r)

            H_ba = tf.transpose(H_ab, perm=[0, 2, 1])
            H = tf.concat(
                [tf.concat([H_aa, H_ab], axis=2),
                 tf.concat([H_ba, H_bb], axis=2)],
                axis=1
            )
            return H

        def _red(prev, cur):
            """
            Reduction operation for hessian computation across observations.

            Every evaluation of the hessian on an observation yields a full
            hessian matrix. This function sums over consecutive evaluations
            of this hessian so that not all separate evaluations have to be
            stored.
            """
            return tf.add(prev, cur)

        params = model_vars.params
        p_shape_a = model_vars.a.shape[0]
        p_shape_b = model_vars.b.shape[0]

        if iterator:
            H = op_utils.map_reduce(
                last_elem=tf.gather(sample_indices, tf.size(sample_indices) - 1),
                data=batched_data,
                map_fn=_assemble_byobs,
                reduce_fn=_red,
                parallel_iterations=pkg_constants.TF_LOOP_PARALLEL_ITERATIONS
            )
        else:
            H = _assemble_byobs(
                idx=sample_indices,
                data=batched_data
            )
        return H

    def byfeature(
            self,
            batched_data,
            sample_indices,
            constraints_loc,
            constraints_scale,
            model_vars: ModelVars,
            dtype
    ):
        """
        Compute the closed-form of the nb_glm model hessian
        by evaluating its terms grouped by features.


        Has three sub-functions which built the specific blocks of the hessian
        and one sub-function which concatenates the blocks into a full hessian.
        """

        def _aa_byfeature(X, design_loc, design_scale, mu, r):
            """
            Compute the mean model diagonal block of the
            closed form hessian of nb_glm model by feature across observation.

            :param X: tf.tensor observations x features
                Observation by observation and feature.
            :param mu: tf.tensor observations x features
                Value of mean model by observation and feature.
            :param r: tf.tensor observations x features
                Value of dispersion model by observation and feature.
            """
            const = _coef_invariant_aa(  # [observations x features=1]
                X=X,
                mu=mu,
                r=r,
            )
            # The second dimension of const is only one element long,
            # this was a feature before but is no recycled into coefficients.
            # const = tf.broadcast_to(const, shape=design_loc.shape)  # [observations, coefficients]
            Hblock = tf.matmul(  # [coefficients, coefficients]
                tf.transpose(design_loc),  # [coefficients, observations]
                tf.multiply(design_loc, const)  # [observations, coefficients]
            )
            return Hblock

        def _bb_byfeature(X, design_loc, design_scale, mu, r):
            """
            Compute the dispersion model diagonal block of the
            closed form hessian of nb_glm model by feature across observation.
            """
            const = _coef_invariant_bb(  # [observations x features=1]
                X=X,
                mu=mu,
                r=r,
            )
            # The second dimension of const is only one element long,
            # this was a feature before but is no recycled into coefficients.
            # const = tf.broadcast_to(const, shape=design_scale.shape)  # [observations, coefficients]
            Hblock = tf.matmul(  # [coefficients, coefficients]
                tf.transpose(design_scale),  # [coefficients, observations]
                tf.multiply(design_scale, const)  # [observations, coefficients]
            )
            return Hblock

        def _ab_byfeature(X, design_loc, design_scale, mu, r):
            """
            Compute the mean-dispersion model off-diagonal block of the
            closed form hessian of nb_glm model by feature across observation.

            Note that there are two blocks of the same size which can
            be compute from each other with a transpose operation as
            the hessian is symmetric.
            """
            const = _coef_invariant_ab(  # [observations x features=1]
                X=X,
                mu=mu,
                r=r,
            )
            # The second dimension of const is only one element long,
            # this was a feature before but is no recycled into coefficients_scale.
            # const = tf.broadcast_to(const, shape=design_scale.shape)  # [observations, coefficients_scale]
            Hblock = tf.matmul(  # [coefficients_loc, coefficients_scale]
                tf.transpose(design_loc),  # [coefficients_loc, observations]
                tf.multiply(design_scale, const)  # [observations, coefficients_scale]
            )
            return Hblock

        def _map(idx, data):
            def _assemble_byfeature(data):
                """
                Assemble hessian of a single feature.

                :param data: tuple
                    Containing the following parameters:
                    - X_t: tf.tensor observations x features .T
                        Observation by observation and feature.
                    - size_factors_t: tf.tensor observations x features .T
                        Model size factors by observation and feature.
                    - params_t: tf.tensor features x coefficients .T
                        Estimated model variables.
                """
                X_t, size_factors_t, params_t = data
                X = tf.transpose(X_t)
                size_factors = tf.transpose(size_factors_t)
                params = tf.transpose(params_t)  # design_params x features
                a_split, b_split = tf.split(params, tf.TensorShape([p_shape_a, p_shape_b]))

                model = BasicModelGraph(
                    X=X,
                    design_loc=design_loc,
                    design_scale=design_scale,
                    constraints_loc=constraints_loc,
                    constraints_scale=constraints_scale,
                    a=a_split,
                    b=b_split,
                    dtype=dtype,
                    size_factors=size_factors
                )
                mu = model.mu
                r = model.r

                H_aa = _aa_byfeature(X=X, design_loc=design_loc, design_scale=design_scale, mu=mu, r=r)
                H_bb = _bb_byfeature(X=X, design_loc=design_loc, design_scale=design_scale, mu=mu, r=r)
                H_ab = _ab_byfeature(X=X, design_loc=design_loc, design_scale=design_scale, mu=mu, r=r)
                H_ba = tf.transpose(H_ab, perm=[1, 0])

                H = tf.concat(
                    [tf.concat([H_aa, H_ab], axis=1),
                     tf.concat([H_ba, H_bb], axis=1)],
                    axis=0
                )
                return [H]

            X, design_loc, design_scale, size_factors = data
            X_t = tf.transpose(tf.expand_dims(X, axis=0), perm=[2, 0, 1])
            size_factors_t = tf.transpose(tf.expand_dims(size_factors, axis=0), perm=[2, 0, 1])
            params_t = tf.transpose(tf.expand_dims(params, axis=0), perm=[2, 0, 1])

            H = tf.map_fn(
                fn=_assemble_byfeature,
                elems=(X_t, size_factors_t, params_t),
                dtype=[dtype],
                parallel_iterations=pkg_constants.TF_LOOP_PARALLEL_ITERATIONS
            )

            return H

        def _red(prev, cur):
            return [tf.add(p, c) for p, c in zip(prev, cur)]

        params = model_vars.params
        p_shape_a = model_vars.a.shape[0]
        p_shape_b = model_vars.b.shape[0]

        H = op_utils.map_reduce(
            last_elem=tf.gather(sample_indices, tf.size(sample_indices) - 1),
            data=batched_data,
            map_fn=_map,
            reduce_fn=_red,
            parallel_iterations=1,
        )
        H = H[0]
        return H

    def tf_byfeature(
            self,
            batched_data,
            sample_indices,
            constraints_loc,
            constraints_scale,
            model_vars: ModelVars,
            dtype
    ) -> List[tf.Tensor]:
        """
        Compute hessians via tf.hessian for all gene-wise models separately.

        Contains three functions:

            - feature_wises_batch():
            a function that computes all hessians for a given batch
            of data by distributing the computation across features.
            - hessian_map():
            a function that unpacks the data from the iterator to run
            feature_wises_batch.
            - hessian_red():
            a function that performs the reduction of the hessians across hessians
            into a single hessian during the iteration over batches.
        """

        def feature_wises_batch(
                X,
                design_loc,
                design_scale,
                constraints_loc,
                constraints_scale,
                params,
                p_shape_a,
                p_shape_b,
                dtype,
                size_factors=None
        ) -> List[tf.Tensor]:
            """
            Compute hessians via tf.hessian for all gene-wise models separately
            for a given batch of data.
            """
            dtype = X.dtype

            # Hessian computation will be mapped across genes/features.
            # The map function maps across dimension zero, the slices have to
            # be 2D tensors to fit into BasicModelGraph, accordingly,
            # X, size_factors and params have to be reshaped to have genes in the first dimension
            # and cells or parameters with an extra padding dimension in the second
            # and third dimension. Note that size_factors is not a 1xobservations array
            # but is implicitly broadcasted to observations x features in Estimator.
            X_t = tf.transpose(tf.expand_dims(X, axis=0), perm=[2, 0, 1])
            size_factors_t = tf.transpose(tf.expand_dims(size_factors, axis=0), perm=[2, 0, 1])
            params_t = tf.transpose(tf.expand_dims(params, axis=0), perm=[2, 0, 1])

            def hessian(data):
                """ Helper function that computes hessian for a given gene.

                :param data: tuple (X_t, size_factors_t, params_t)
                """
                # Extract input data:
                X_t, size_factors_t, params_t = data
                size_factors = tf.transpose(size_factors_t)  # observations x features
                X = tf.transpose(X_t)  # observations x features
                params = tf.transpose(params_t)  # design_params x features

                a_split, b_split = tf.split(params, tf.TensorShape([p_shape_a, p_shape_b]))

                # Define the model graph based on which the likelihood is evaluated
                # which which the hessian is computed:
                model = BasicModelGraph(
                    X=X,
                    design_loc=design_loc,
                    design_scale=design_scale,
                    constraints_loc=constraints_loc,
                    constraints_scale=constraints_scale,
                    a=a_split,
                    b=b_split,
                    dtype=dtype,
                    size_factors=size_factors
                )

                # Compute the hessian of the model of the given gene:
                H = tf.hessians(- model.log_likelihood, params)
                return H

            # Map hessian computation across genes
            H = tf.map_fn(
                fn=hessian,
                elems=(X_t, size_factors_t, params_t),
                dtype=[dtype],  # hessians of [a, b]
                parallel_iterations=pkg_constants.TF_LOOP_PARALLEL_ITERATIONS
            )

            H = [tf.squeeze(tf.squeeze(tf.stack(h), axis=2), axis=3) for h in H]

            return H

        def _map(idx, data):
            X, design_loc, design_scale, size_factors = data
            return feature_wises_batch(
                X=X,
                design_loc=design_loc,
                design_scale=design_scale,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                params=model_vars.params,
                p_shape_a=model_vars.a.shape[0],
                p_shape_b=model_vars.b.shape[0],
                dtype=dtype,
                size_factors=size_factors
            )

        def _red(prev, cur):
            return [tf.add(p, c) for p, c in zip(prev, cur)]

        H = op_utils.map_reduce(
            last_elem=tf.gather(sample_indices, tf.size(sample_indices) - 1),
            data=batched_data,
            map_fn=_map,
            reduce_fn=_red,
            parallel_iterations=1,
        )
        return H[0]
