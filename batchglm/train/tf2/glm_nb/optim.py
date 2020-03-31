import tensorflow as tf
import numpy as np
from .external import IRLS, pkg_constants

class IRLS_LS(IRLS):

    def __init__(self, dtype, trusted_region_mode, model, name, n_obs, max_iter):

        super(IRLS_LS, self).__init__(
            dtype=dtype,
            trusted_region_mode=trusted_region_mode,
            model=model,
            name=name,
            n_obs=n_obs)

        self.max_iter = max_iter

        if name.startswith('irls_gd'):
            self.update_b_func = self.update_b_gd
            if trusted_region_mode:
                n_features = self.model.model_vars.n_features
                self.tr_radius_b = tf.Variable(
                    np.zeros(shape=[n_features]) + pkg_constants.TRUST_REGION_RADIUS_INIT_SCALE,
                    dtype=self._dtype, trainable=False)

        elif name in ['irls_ar_tr', 'irls_ar']:
            self.update_b_func = self.update_b_armijio

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

    def perform_parameter_update(self, inputs, compute_a=True, compute_b=True, batch_features=False, is_batched=False):

        assert compute_a ^ compute_b, \
            "IRLSLS computes either loc or scale model updates, not both nor none at the same time."

        if compute_a:
            super(IRLS_LS, self).perform_parameter_update(
                inputs, compute_a, compute_b, batch_features, is_batched)
        else:
            all_features_converged = False
            i = 0
            while(not all_features_converged and i < self.max_iter):
                all_features_converged = self.update_b_func(inputs, batch_features, is_batched)
                i += 1
                print(i)

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

        x_batches, log_probs, _, jac_b, _, _ = inputs

        update_b = tf.transpose(jac_b)
        if not self.trusted_region_mode:
            update = self._pad_updates(
                update_raw=update_b,
                compute_a=False,
                compute_b=True
            )
            if batch_features:
                indices = tf.where(self.model.model_vars.remaining_features)
                update_var = tf.transpose(
                    tf.scatter_nd(
                        indices,
                        tf.transpose(update),
                        shape=(self.model.model_vars.n_features, update.get_shape()[0])
                    )
                )
            else:
                update_var = update
            self.model.params.assign_sub(update_var)

        else:
            if batch_features:
                radius_container = tf.boolean_mask(
                    tensor=self.tr_radius_b,
                    mask=self.model.model_vars.remaining_features)
            else:
                radius_container = self.tr_radius_b
            print(update_b.shape)
            print(radius_container.shape)
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
            self._trust_region_ops(
                x_batches=x_batches,
                log_probs=log_probs,
                proposed_vector=tr_update_b,
                proposed_gain=None,  # TODO remove completely, not needed any longer
                compute_a=False,
                compute_b=True,
                batch_features=batch_features,
                is_batched=is_batched
            )

        return False

    def update_b_ar(self, inputs, batch_features, is_batched):

        raise NotImplementedError('Armijio line search not implemented yet.')
        """
        x_batches = inputs[0]
        proposed_vector = self._perform_trial_update()
        self._check_and_apply_update(x_batches, proposed_vector, batch_features)

        return None
        """

    def _check_and_apply_update(
        self,
        x_batches,
        proposed_vector,
        batch_features,
    ):
        eta0 = tf.constant(pkg_constants.TRUST_REGION_ETA0, dtype=self._dtype)
        """
        Current likelihood refers to the likelihood that has been calculated in the last model call.
        We are always evaluating on the full model, so if we train on the batched model (is_batched),
        current likelihood needs to be calculated on the full model using the same model state as
        used in the last model call. Moreover, if this update is conducted separately for loc
        (compute_a) and scale (compute_b), current likelihood always needs to be recalculated when
        updating the scale params since the location params changed in the location update before.
        This is only true if the location params are updated before the scale params however!
        """

        for i, x_batch in enumerate(x_batches):
            log_likelihood = self.model.calc_ll([*x_batch])[0]
            if i == 0:
                current_likelihood = log_likelihood
            else:
                current_likelihood = tf.math.add(current_likelihood, log_likelihood)

        current_likelihood = self._norm_neg_log_likelihood(current_likelihood)

        """
        The new likelihood is calculated on the full model now, after updating the parameters using
        the proposed vector:
        """
        original_params_copy = tf.identity(self.model.params_copy)
        self.model.params_copy.assign_sub(proposed_vector)
        for i, x_batch in enumerate(x_batches):
            log_likelihood = self.model.calc_ll([*x_batch])[0]
            if i == 0:
                new_likelihood = log_likelihood
            else:
                new_likelihood += log_likelihood
        new_likelihood = self._norm_neg_log_likelihood(new_likelihood)

        """
        delta_f_actual shows the difference between the log likelihoods before and after the proposed
        update of parameters. It is > 0 if the new likelihood is greater than the old.
        """
        delta_f_actual = tf.math.subtract(current_likelihood, new_likelihood)

        update_theta = delta_f_actual > eta0
        self.model.params_copy.assign(tf.where(update_theta, self.model.params_copy, original_params_copy))

        if batch_features:
            n_features = self.model.model_vars.n_features
            indices = tf.where(self.model.model_vars.remaining_features)
            update_theta = tf.scatter_nd(indices, update_theta, shape=(n_features,))

        self.model.model_vars.updated_b = update_theta.numpy()
