from .external import pkg_constants
import tensorflow as tf
from .external import OptimizerBase
import abc
import numpy as np


class SecondOrderOptim(OptimizerBase, metaclass=abc.ABCMeta):

    """
    Superclass for NR and IRLS
    """

    def _norm_neg_log_likelihood(self, log_probs):
        return - log_probs / self.n_obs

    def _resource_apply_dense(self, grad, handle, apply_state=None):

        update_op = handle.assign_add(grad, read_value=False)

        return update_op

    def _resource_apply_sparse(self, grad, handle, apply_state=None):

        raise NotImplementedError('Applying SparseTensor currently not possible.')

    def get_config(self):

        config = {"name": "SOO"}
        return config

    def _create_slots(self, var_list):

        self.add_slot(var_list[0], 'mu_r')

    def gett1t2(self):
        t1 = tf.constant(pkg_constants.TRUST_REGION_T1, dtype=self._dtype)
        t2 = tf.constant(pkg_constants.TRUST_REGION_T2, dtype=self._dtype)
        return t1, t2

    def _trust_region_ops(
            self,
            x_batches,
            log_probs,
            proposed_vector,
            proposed_gain,
            compute_a,
            compute_b,
            batch_features,
            is_batched
    ):
        # Load hyper-parameters:
        # assert pkg_constants.TRUST_REGION_ETA0 < pkg_constants.TRUST_REGION_ETA1, \
        #    "eta0 must be smaller than eta1"
        # assert pkg_constants.TRUST_REGION_ETA1 <= pkg_constants.TRUST_REGION_ETA2, \
        #    "eta1 must be smaller than or equal to eta2"
        # assert pkg_constants.TRUST_REGION_T1 <= 1, "t1 must be smaller than 1"
        # assert pkg_constants.TRUST_REGION_T2 >= 1, "t1 must be larger than 1"
        # Set trust region hyper-parameters
        eta0 = tf.constant(pkg_constants.TRUST_REGION_ETA0, dtype=self._dtype)
        # eta2 = tf.constant(pkg_constants.TRUST_REGION_ETA2, dtype=self._dtype)
        t1, t2 = self.gett1t2()

        upper_bound = tf.constant(pkg_constants.TRUST_REGION_UPPER_BOUND, dtype=self._dtype)

        # Phase I: Perform a trial update.
        # Propose parameter update:
        """
        Current likelihood refers to the likelihood that has been calculated in the last model call.
        We are always evaluating on the full model, so if we train on the batched model (is_batched),
        current likelihood needs to be calculated on the full model using the same model state as
        used in the last model call. Moreover, if this update is conducted separately for loc
        (compute_a) and scale (compute_b), current likelihood always needs to be recalculated when
        updating the scale params since the location params changed in the location update before.
        This is only true if the location params are updated before the scale params however!
        """
        current_likelihood = log_probs
        if is_batched or compute_b and not compute_a:
            for i, x_batch in enumerate(x_batches):
                log_likelihood = self.model.calc_ll([*x_batch])[0]
                current_likelihood = log_likelihood if i == 0 else tf.math.add(current_likelihood, log_likelihood)

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

        """
        If we use feature batching, the individual vector indices need to be spread out to the full
        feature space by adding columns corresponding to positions of converged (non calculated)
        features.
        """

        # Compute parameter updates.g
        update_theta = delta_f_actual > eta0
        self.model.params_copy.assign(tf.where(update_theta, self.model.params_copy, original_params_copy))

        #update_theta_numeric = tf.expand_dims(tf.cast(update_theta, self._dtype), axis=0)
        #keep_theta_numeric = tf.ones_like(update_theta_numeric) - update_theta_numeric

        decrease_radius = tf.math.logical_not(update_theta)
        increase_radius = update_theta
        if batch_features:
            n_features = self.model.model_vars.n_features
            indices = tf.where(self.model.model_vars.remaining_features)
            decrease_radius = tf.scatter_nd(indices, decrease_radius, shape=(n_features,))
            increase_radius = tf.scatter_nd(indices, update_theta, shape=(n_features,))

        if compute_b and not compute_a:
            self.model.model_vars.updated_b = increase_radius.numpy()
        else:
            self.model.model_vars.updated = increase_radius.numpy()

        # Update trusted region accordingly:

        keep_radius = tf.logical_and(tf.logical_not(decrease_radius),
                                     tf.logical_not(increase_radius))
        radius_update = tf.add_n([
            tf.multiply(t1, tf.cast(decrease_radius, self._dtype)),
            tf.multiply(t2, tf.cast(increase_radius, self._dtype)),
            tf.multiply(tf.ones_like(t1), tf.cast(keep_radius, self._dtype))
        ])

        if compute_b and not compute_a:
            tr_radius = self.tr_radius_b
        else:
            tr_radius = self.tr_radius

        radius_new = tf.minimum(tf.multiply(tr_radius, radius_update), upper_bound)
        tr_radius.assign(radius_new)

        return update_theta

    def __init__(self, dtype: tf.dtypes.DType, trusted_region_mode: bool, model: tf.keras.Model, name: str, n_obs: int):

        super(SecondOrderOptim, self).__init__(name)

        self.model = model
        self._dtype = dtype
        self.n_obs = tf.cast(n_obs, dtype=self._dtype)
        self.trusted_region_mode = trusted_region_mode

        if trusted_region_mode:
            n_features = self.model.model_vars.n_features
            self.tr_radius = tf.Variable(
                np.zeros(shape=[n_features]) + pkg_constants.TRUST_REGION_RADIUS_INIT,
                dtype=self._dtype, trainable=False)

        else:
            self.tr_radius = tf.Variable(np.array([np.inf]), dtype=self._dtype, trainable=False)

    @abc.abstractmethod
    def perform_parameter_update(self, inputs, compute_a=True, compute_b=True, batch_features=False, is_batched=False):
        pass

    def _newton_type_update(self, lhs, rhs, psd=False):

        new_rhs = tf.expand_dims(rhs, axis=-1)
        res = tf.linalg.lstsq(lhs, new_rhs, fast=psd)
        delta_t = tf.squeeze(res, axis=-1)
        update_tensor = tf.transpose(delta_t)
        return update_tensor

    def _pad_updates(
            self,
            update_raw,
            compute_a,
            compute_b
    ):
        # Pad update vectors to receive update tensors that match
        # the shape of model_vars.params.
        if compute_a:
            if compute_b:
                netwon_type_update = update_raw
            else:
                netwon_type_update = tf.concat([
                    update_raw,
                    tf.zeros(shape=(self.model.model_vars.b_var.get_shape()[0], update_raw.get_shape()[1]),
                             dtype=self._dtype)
                ], axis=0)

        elif compute_b:
            netwon_type_update = tf.concat([
                tf.zeros(shape=(self.model.model_vars.a_var.get_shape()[0], update_raw.get_shape()[1]),
                         dtype=self._dtype),
                update_raw
            ], axis=0)

        else:
            raise ValueError("No training necessary")

        return netwon_type_update

    @staticmethod
    def _calc_update_magnitudes(update_raw):
        update_magnitude_sq = tf.reduce_sum(tf.square(update_raw), axis=0)
        update_magnitude = tf.where(
            condition=update_magnitude_sq > 0,
            x=tf.sqrt(update_magnitude_sq),
            y=tf.zeros_like(update_magnitude_sq)
        )
        update_magnitude_inv = tf.where(
            condition=update_magnitude > 0,
            x=tf.divide(
                tf.ones_like(update_magnitude),
                update_magnitude
            ),
            y=tf.zeros_like(update_magnitude)
        )
        return update_magnitude, update_magnitude_inv

    def _trust_region_update(
            self,
            update_raw,
            radius_container,
    ):
        update_magnitude, update_magnitude_inv = SecondOrderOptim._calc_update_magnitudes(update_raw)
        update_norm = tf.multiply(update_raw, update_magnitude_inv)

        update_scale = tf.minimum(
            radius_container,
            update_magnitude
        )
        proposed_vector = tf.multiply(
            update_norm,
            update_scale
        )

        return proposed_vector

    def normalize_update_magnitude(self, update_magnitude):
        return update_magnitude

    def _trust_region_newton_cost_gain(
            self,
            proposed_vector,
            neg_jac,
            hessian_fim
    ):
        pred_cost_gain = tf.add(
            tf.einsum(
                'ni,in->n',
                neg_jac,
                proposed_vector
            ) / self.n_obs,
            0.5 * tf.einsum(
                'nix,xin->n',
                tf.einsum('inx,nij->njx',
                          tf.expand_dims(proposed_vector, axis=-1),
                          hessian_fim),
                tf.expand_dims(proposed_vector, axis=0)
            ) / tf.square(self.n_obs)
        )
        return pred_cost_gain


class NR(SecondOrderOptim):

    def _get_updates(self, lhs, rhs, compute_a, compute_b):

        update_raw = self._newton_type_update(lhs=lhs, rhs=rhs)
        update = self._pad_updates(update_raw, compute_a, compute_b)

        return update_raw, update

    def perform_parameter_update(self, inputs, compute_a=True, compute_b=True, batch_features=False, is_batched=False):

        x_batches, log_probs, jacobians, hessians = inputs
        if compute_b:
            if not compute_a:
                self.model.model_vars.updated_b = np.repeat(a=False, repeats=self.params.shape[1])  # Initialise to is updated.

        assert (compute_a or compute_b), "Nothing can be trained. Please make sure" \
            "at least one of train_mu and train_r is set to True."

        update_raw, update = self._get_updates(hessians, jacobians, compute_a, compute_b)

        if self.trusted_region_mode:

            if batch_features:
                radius_container = tf.boolean_mask(
                    tensor=self.tr_radius,
                    mask=self.model.model_vars.remaining_features)
            else:
                radius_container = self.tr_radius
            tr_proposed_vector = self._trust_region_update(
                update_raw=update_raw,
                radius_container=radius_container
            )
            tr_pred_cost_gain = self._trust_region_newton_cost_gain(
                proposed_vector=tr_proposed_vector,
                neg_jac=jacobians,
                hessian_fim=hessians
            )

            tr_proposed_vector_pad = self._pad_updates(
                update_raw=tr_proposed_vector,
                compute_a=compute_a,
                compute_b=compute_b
            )

            self._trust_region_ops(
                x_batches=x_batches,
                log_probs=log_probs,
                proposed_vector=tr_proposed_vector_pad,
                proposed_gain=tr_pred_cost_gain,
                compute_a=compute_a,
                compute_b=compute_b,
                batch_features=batch_features,
                is_batched=is_batched
            )

        else:
            self.model.params_copy.assign_sub(update)


class IRLS(SecondOrderOptim):

    def _calc_proposed_vector_and_pred_cost_gain(
            self,
            update_x,
            radius_container,
            neg_jac_x,
            fim_x=None
    ):
        """
        Calculates the proposed vector and predicted cost gain for either mean or scale part.
        :param update_x: tf.tensor coefficients x features ? TODO

        :param radius_container: tf.tensor ? x ? TODO

        :param neg_jac_x: tf.Tensor coefficients x features ? TODO
            Upper (mu part) or lower (r part) of negative jacobian matrix
        :param fim_x
            Upper (mu part) or lower (r part) of Fisher Inverse Matrix.
            Defaults to None, is only needed if gd is False
        :return proposed_vector_x, pred_cost_gain_x
            Returns proposed vector and predicted cost gain after
            trusted region update for either mu or r part, depending on x
        """

        proposed_vector_x = self._trust_region_update(
            update_raw=update_x,
            radius_container=radius_container
        )

        pred_cost_gain_x = self._trust_region_newton_cost_gain(
            proposed_vector=proposed_vector_x,
            neg_jac=neg_jac_x,
            hessian_fim=fim_x
        )

        return proposed_vector_x, pred_cost_gain_x


    def perform_parameter_update(self, inputs, compute_a=True, compute_b=True, batch_features=False, is_batched=False):

        x_batches, log_probs, jac_a, jac_b, fim_a, fim_b = inputs
        if compute_b:
            if not compute_a:
                self.model.model_vars.updated_b = np.repeat(a=False, repeats=self.params.shape[1])  # Initialise to is updated.

        assert (compute_a or compute_b), "Nothing can be trained. Please make sure" \
            "at least one of train_mu and train_r is set to True."

        # Compute a and b model updates separately.
        if compute_a:
            # The FIM of the mean model is guaranteed to be
            # positive semi-definite and can therefore be inverted
            # with the Cholesky decomposition. This information is
            # passed here with psd=True.
            update_a = self._newton_type_update(
                lhs=fim_a,
                rhs=jac_a,
                psd=False
            )
        if compute_b:

            update_b = self._newton_type_update(
                lhs=fim_b,
                rhs=jac_b
            )

        if not self.trusted_region_mode:
            if compute_a:
                if compute_b:
                    update_raw = tf.concat([update_a, update_b], axis=0)
                else:
                    update_raw = update_a
            else:
                update_raw = update_b

            update = self._pad_updates(
                update_raw=update_raw,
                compute_a=compute_a,
                compute_b=compute_b
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
            self.model.params_copy.assign_sub(update_var)

        else:
            # put together update_raw based on proposed vector and cost gain depending on train_r and train_mu
            if batch_features:
                radius_container = tf.boolean_mask(
                    tensor=self.tr_radius,
                    mask=self.model.model_vars.remaining_features)
            else:
                radius_container = self.tr_radius

            if compute_b:
                tr_proposed_vector_b, tr_pred_cost_gain_b = self._calc_proposed_vector_and_pred_cost_gain(
                    update_b, radius_container, jac_b, fim_b)
                if compute_a:
                    tr_proposed_vector_a, tr_pred_cost_gain_a = self._calc_proposed_vector_and_pred_cost_gain(
                        update_a, radius_container, jac_a, fim_a)

                    tr_update_raw = tf.concat([tr_proposed_vector_a, tr_proposed_vector_b], axis=0)
                    tr_pred_cost_gain = tf.add(tr_pred_cost_gain_a, tr_pred_cost_gain_b)
                else:
                    # directly apply output of calc_proposed_vector_and_pred_cost_gain to tr_update_raw
                    # and tr_pred_cost_gain
                    tr_update_raw = tr_proposed_vector_b
                    tr_pred_cost_gain = tr_pred_cost_gain_b
            else:
                # here train_r is False AND train_mu is true, so the output of the function can directly be applied to
                # tr_update_raw and tr_pred_cost_gain, similar to train_r = True and train_mu = False
                tr_update_raw, tr_pred_cost_gain = self._calc_proposed_vector_and_pred_cost_gain(
                    update_a, radius_container, jac_a, fim_a)

            tr_update = self._pad_updates(
                update_raw=tr_update_raw,
                compute_a=compute_a,
                compute_b=compute_b
            )

            # perform update
            self._trust_region_ops(
                x_batches=x_batches,
                log_probs=log_probs,
                proposed_vector=tr_update,
                proposed_gain=tr_pred_cost_gain,
                compute_a=compute_a,
                compute_b=compute_b,
                batch_features=batch_features,
                is_batched=is_batched
            )

    def calc_delta_f_actual(self, current_likelihood, new_likelihood, jacobian):
        eta1 = tf.constant(pkg_constants.TRUST_REGION_ETA1, dtype=self._dtype)
        return
