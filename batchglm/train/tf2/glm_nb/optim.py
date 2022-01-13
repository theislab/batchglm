import tensorflow as tf
import numpy as np
from .external import IRLS, pkg_constants

class IRLS_LS(IRLS):

    def __init__(self, dtype, tr_mode, model, name, n_obs, intercept_scale):

        parent_tr_mode = False
        self.tr_mode_b = False
        if name.startswith('irls_tr'):
            parent_tr_mode = True  # for loc model
        if name in ['irls_tr_gd_tr', 'irls_gd_tr', 'irls_gd', 'irls_tr_gd']:
            self.update_b_func = self.update_b_gd
        elif name in ['irls_ar', 'irls_tr_ar']:
            assert intercept_scale, "Line search (armijo) is only available" \
                "for scale models with a single coefficient (intercept scale)."
            self.update_b_func = self.update_b_ar
        else:
            assert False, "Unrecognized method for optimization given."
        super(IRLS_LS, self).__init__(
            dtype=dtype,
            tr_mode=parent_tr_mode,
            model=model,
            name=name,
            n_obs=n_obs)

        if tr_mode:
            n_features = self.model.model_vars.n_features
            self.tr_radius_b = tf.Variable(
                np.zeros(shape=[n_features]) + pkg_constants.TRUST_REGION_RADIUS_INIT_SCALE,
                dtype=self._dtype, trainable=False)
            self.tr_mode_b = True

    def _trust_region_linear_cost_gain(
            self,
            proposed_vector,
            neg_jac
    ):
        pred_cost_gain = tf.reduce_sum(tf.multiply(
            proposed_vector,
            tf.transpose(neg_jac)
        ), axis=0)
        return pred_cost_gain

    def perform_parameter_update(
        self,
        inputs,
        compute_a=True,
        compute_b=True,
        batch_features=False,
        is_batched=False,
        maxiter=1
    ):
        assert compute_a ^ compute_b, \
            "IRLS_LS computes either loc or scale model updates, not both nor none at the same time."

        if compute_a:
            super(IRLS_LS, self).perform_parameter_update(
                inputs, compute_a, compute_b, batch_features, is_batched)
        else:
            # global_step = tf.zeros_like(self.model.model_vars.remaining_features)
            results = inputs[1:4]
            x_batches = inputs[0]
            iteration = 0
            not_converged = np.zeros_like(self.model.model_vars.remaining_features)
            updated_b = np.zeros_like(self.model.model_vars.updated_b)
            while True:
                iteration += 1
                step = self.update_b_func([x_batches, *results], batch_features, is_batched)
                not_converged = tf.abs(step).numpy() > pkg_constants.XTOL_BY_FEATURE_SCALE
                updated_b |= self.model.model_vars.updated_b
                if not tf.reduce_any(not_converged) or iteration == maxiter:
                    break
                for i, x_batch in enumerate(inputs[0]):
                    results = self.model.calc_jacobians(x_batch, concat=False, compute_a=False) if i == 0 else \
                        [tf.math.add(results[i], x) for
                         i, x in enumerate(self.model.calc_jacobians(x_batch, concat=False, compute_a=False))]
            self.model.model_vars.updated_b = updated_b

    def gett1t2(self):
        t1 = tf.constant(pkg_constants.TRUST_REGIONT_T1_IRLS_GD_TR_SCALE, dtype=self._dtype)
        t2 = tf.constant(pkg_constants.TRUST_REGIONT_T2_IRLS_GD_TR_SCALE, dtype=self._dtype)
        return t1, t2

    def _trust_region_update_b(
            self,
            update_raw,
            radius_container,
    ):
        update_magnitude, update_magnitude_inv = IRLS_LS._calc_update_magnitudes(update_raw)
        update_norm = tf.multiply(update_raw, update_magnitude_inv)

        update_magnitude = update_magnitude / self.n_obs * radius_container

        update_scale = tf.minimum(
            radius_container,
            update_magnitude
        )
        proposed_vector = tf.multiply(
            update_norm,
            update_scale
        )

        return proposed_vector
    def update_b_gd(self, inputs, batch_features, is_batched):

        x_batches, log_probs, _, jac_b = inputs

        update_b = tf.transpose(jac_b)
        if not self.tr_mode_b:
            update = self._pad_updates(
                update_raw=update_b,
                compute_a=False,
                compute_b=True
            )

            update_theta = self._trial_update(
                x_batches=x_batches,
                log_probs=log_probs,
                proposed_vector=update,
                is_batched=is_batched,
                compute_a=False,
                compute_b=True
            )
            self.model.params_copy.assign_sub(update)

            return tf.where(update_theta, update, tf.zeros_like(update))

        else:
            if batch_features:
                radius_container = tf.boolean_mask(
                    tensor=self.tr_radius_b,
                    mask=self.model.model_vars.remaining_features)
            else:
                radius_container = self.tr_radius_b

            tr_proposed_vector_b = self._trust_region_update_b(
                update_raw=update_b,
                radius_container=radius_container
            )

            tr_update_b = self._pad_updates(
                update_raw=tr_proposed_vector_b,
                compute_a=False,
                compute_b=True
            )

            # perform update
            update_theta = self._trial_update(
                x_batches=x_batches,
                log_probs=log_probs,
                proposed_vector=tr_update_b,
                is_batched=is_batched,
                compute_a=False,
                compute_b=True)
            self._trust_region_ops(
                proposed_vector=tr_update_b,
                compute_a=False,
                compute_b=True,
                batch_features=batch_features,
                update_theta=update_theta)

            return tf.where(update_theta, tr_proposed_vector_b, tf.zeros_like(tr_proposed_vector_b))

    def update_b_ar(self, inputs, batch_features, is_batched, alpha0=None):

        x_batches, log_probs, _, jac_b = inputs
        jac_b = tf.reshape(jac_b, [jac_b.shape[0]])
        direction = -tf.sign(jac_b)
        derphi0 = jac_b / self.n_obs
        if alpha0 is None:
            alpha0 = tf.ones_like(jac_b) * pkg_constants.ALPHA0
        original_params_b_copy = self.model.params_copy[-1]

        def phi(alpha):
            multiplier = tf.multiply(alpha, direction)
            new_scale_params = tf.add(original_params_b_copy, multiplier)
            self.model.params_copy[-1].assign(new_scale_params)
            new_likelihood = None
            for i, x_batch in enumerate(x_batches):
                log_likelihood = self.model.calc_ll([*x_batch])[0]
                new_likelihood = log_likelihood if i == 0 else \
                    tf.math.add(new_likelihood, log_likelihood)
            new_likelihood = self._norm_neg_log_likelihood(new_likelihood)
            return new_likelihood

        current_likelihood = self._norm_neg_log_likelihood(log_probs)
        new_likelihood = phi(alpha0)
        beneficial = self.wolfe1(current_likelihood, new_likelihood, alpha0, derphi0)

        if tf.reduce_all(beneficial):  # are all beneficial?
            updated = beneficial
            if batch_features:
                n_features = self.model.model_vars.n_features
                indices = tf.where(self.model.model_vars.remaining_features)
                updated = tf.scatter_nd(indices, beneficial, shape=(n_features,))
            self.model.model_vars.updated_b = updated
            return tf.multiply(alpha0, direction)

        divisor = new_likelihood - current_likelihood - derphi0 * alpha0
        alpha1 = tf.negative(derphi0) * alpha0**2 / 2 / divisor
        alpha1 = tf.where(beneficial, alpha0, alpha1)
        new_likelihood2 = phi(alpha1)
        beneficial = self.wolfe1(current_likelihood, new_likelihood2, alpha1, derphi0)
        if tf.reduce_all(beneficial):
            updated = beneficial
            if batch_features:
                n_features = self.model.model_vars.n_features
                indices = tf.where(self.model.model_vars.remaining_features)
                updated = tf.scatter_nd(indices, beneficial, shape=(n_features,))
            self.model.model_vars.updated_b = updated
            return tf.multiply(alpha1, direction)

        if not tf.reduce_any(alpha1 > pkg_constants.XTOL_BY_FEATURE_SCALE):
            # catch in case it doesn't enter the loop.
            new_scale_params = tf.where(beneficial, self.model.params_copy[-1], original_params_b_copy)
            self.model.params_copy[-1].assign(new_scale_params)
            self.model.model_vars.updated_b = np.ones_like(self.model.model_vars.updated_b)
            return tf.multiply(alpha1, direction)

        while tf.reduce_any(alpha1 > pkg_constants.XTOL_BY_FEATURE_SCALE):

            factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
            a = alpha0**2 * (new_likelihood2 - current_likelihood - derphi0 * alpha1) - \
                alpha1**2 * (new_likelihood - current_likelihood - derphi0 * alpha0)
            a = a / factor

            b = -alpha0**3 * (new_likelihood2 - current_likelihood - derphi0 * alpha1) + \
                alpha1**3 * (new_likelihood - current_likelihood - derphi0 * alpha0)
            b = b / factor

            alpha2 = (-b + tf.sqrt(tf.abs(tf.square(b) - 3 * a * derphi0))) / (3 * a)
            alpha2 = tf.where(beneficial, alpha1, alpha2)
            idx_to_clip = tf.logical_or(tf.math.is_nan(alpha2), alpha2 < 0)
            alpha2 = tf.where(idx_to_clip, tf.zeros_like(alpha2), alpha2)
            new_likelihood3 = phi(alpha2)
            beneficial = self.wolfe1(current_likelihood, new_likelihood3, alpha2, derphi0)

            if tf.reduce_all(beneficial):
                updated = beneficial
                if batch_features:
                    n_features = self.model.model_vars.n_features
                    indices = tf.where(self.model.model_vars.remaining_features)
                    updated = tf.scatter_nd(indices, beneficial, shape=(n_features,))
                self.model.model_vars.updated_b = updated
                return tf.multiply(alpha2, direction)

            step_diff_greater_half_alpha1 = (alpha1 - alpha2) > alpha1 / 2
            ratio = (1 - alpha2/alpha1) < 0.96
            set_back = tf.logical_or(step_diff_greater_half_alpha1, ratio)
            alpha2 = tf.where(set_back, alpha1 / 2, alpha2)
            alpha2 = tf.where(tf.logical_or(tf.math.is_nan(alpha2), alpha2 < 0), tf.zeros_like(alpha2), alpha2)

            alpha0 = alpha1
            alpha1 = alpha2
            new_likelihood = new_likelihood2
            new_likelihood2 = new_likelihood3

        new_scale_params = tf.where(beneficial, self.model.params_copy[-1], original_params_b_copy)
        self.model.params_copy[-1].assign(new_scale_params)
        updated = beneficial
        if batch_features:
            n_features = self.model.model_vars.n_features
            indices = tf.where(self.model.model_vars.remaining_features)
            updated = tf.scatter_nd(indices, beneficial, shape=(n_features,))
        self.model.model_vars.updated_b = np.ones_like(self.model.model_vars.updated_b)
        return tf.multiply(alpha2, direction)

    def wolfe1(self, current_likelihood, new_likelihood, alpha, jacobian):
        """Checks if an update satisfies the first wolfe condition by returning the difference
        to the previous likelihood."""
        c1 = tf.constant(pkg_constants.WOLFE_C1, dtype=self._dtype)
        limit = tf.add(current_likelihood, tf.multiply(tf.multiply(c1, alpha), jacobian))
        return new_likelihood < limit
