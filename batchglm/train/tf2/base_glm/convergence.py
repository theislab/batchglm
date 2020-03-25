import numpy as np
import tensorflow as tf
from .external import pkg_constants


class ConvergenceCalculator:
    """Wrapper object containing all necessary methods to calculate convergences based on change
    in likelihood, gradient and parameters."""

    def __init__(self, estimator, last_ll: np.ndarray):
        self.estimator = estimator
        self.last_params = estimator.model.params_copy.numpy()
        self.last_ll = last_ll
        self.previous_number_converged = 0
        self.calc_separated = self.estimator.irls_algo and self.estimator._train_scale

    def calculate_convergence(self, results, jac_normalization, optimizer_object, batch_features):
        """Calculates convergence based on change in likelihood, gradient and parameters."""

        features_updated = self.estimator.model.model_vars.updated
        converged_a = self.estimator.model.model_vars.converged
        not_converged_a = ~ converged_a
        if self.calc_separated:
            features_updated_b = self.estimator.model.model_vars.updated_b
            converged_b = self.estimator.model.model_vars.converged_b
            not_converged_b = ~ converged_b

        n_features = self.estimator.input_data.num_features

        ###########################################################
        # FIRST PART: Retrieve and manipulate ll, grads and params.
        ####
        if self.estimator.irls_algo:
            grad_numpy = tf.abs(tf.concat((results[1], results[2]), axis=1))
        elif self.estimator.nr_algo:
            grad_numpy = tf.abs(results[1])
        else:
            grad_numpy = tf.abs(tf.transpose(results[1]))
        new_ll = tf.negative(tf.divide(results[0], self.estimator.input_data.num_observations))
        new_params = self.estimator.model.params_copy

        if batch_features:
            # map columns of ll to full feature space
            remaining_features = self.estimator.model.model_vars.remaining_features
            indices = tf.where(remaining_features)
            updated_lls = tf.scatter_nd(indices, new_ll, shape=[n_features])
            # fill the added columns with previous ll
            new_ll = tf.where(remaining_features, updated_lls, self.last_ll)

            # fill added columns with the gradients from previous runs.
            grad_numpy = tf.scatter_nd(
                indices,
                grad_numpy,
                shape=(n_features, self.estimator.model.params.get_shape()[0])
            )
            # TODO: added columns are zero here, does that matter?

            # map columns of params to full feature space
            new_params = tf.transpose(
                tf.scatter_nd(
                    indices,
                    tf.transpose(new_params),
                    shape=(self.estimator.model.params.shape[1], self.estimator.model.params.shape[0])
                )
            )
            # TODO: added columns are zero here, does that matter?

        grad_numpy = grad_numpy.numpy()
        new_params = new_params.numpy()
        new_ll = new_ll.numpy()

        ###########################################################
        # SECOND PART: Calculate ll convergence.
        ####

        # Get all converged features due to change in ll < LLTOL_BY_FEATURE
        # IMPORTANT: we need to ensure they have also been updated, otherwise ll_prev = ll_current!
        ll_difference = np.abs(self.last_ll - new_ll) / self.last_ll
        # print('ll_diff: ', ll_difference[0])
        # print(self.estimator.model.model_vars.converged[0], self.estimator.model.model_vars.updated[0])
        # print(self.estimator.model.model_vars.converged_b[0], self.estimator.model.model_vars.updated_b[0])
        ll_converged = ll_difference < pkg_constants.LLTOL_BY_FEATURE

        ll_converged_a = ll_converged & features_updated
        epoch_ll_converged_a = not_converged_a & ll_converged_a  # formerly known as converged_f

        if self.calc_separated:
            ll_converged_b = ll_converged & features_updated_b
            epoch_ll_converged_b = not_converged_b & ll_converged_b  # formerly known as converged_f

        ###########################################################
        # THIRD PART: calculate grad convergence.
        ####
        grad_loc = np.sum(grad_numpy[:, self.estimator.model.model_vars.idx_train_loc], axis=1)
        grad_norm_loc = grad_loc / jac_normalization
        grad_scale = np.sum(grad_numpy[:, self.estimator.model.model_vars.idx_train_scale], axis=1)
        grad_norm_scale = grad_scale / jac_normalization

        grad_norm_loc_converged = grad_norm_loc < pkg_constants.GTOL_BY_FEATURE_LOC
        grad_norm_scale_converged = grad_norm_scale < pkg_constants.GTOL_BY_FEATURE_SCALE
        if self.calc_separated:
            grad_converged_a = grad_norm_loc_converged & features_updated
            grad_converged_b = grad_norm_scale_converged & features_updated_b
            epoch_grad_converged_b = not_converged_b & grad_converged_b  # formerly known as converged_g

        else:
            grad_converged_a = grad_norm_loc_converged & grad_norm_scale_converged & features_updated
        epoch_grad_converged_a = not_converged_a & grad_converged_a  # formerly known as converged_g
        # print('grad: ', grad_norm_loc[0], grad_norm_scale[0])

        ###########################################################
        # Fourth PART: calculate parameter step convergence.
        ####
        x_step_a, x_step_b = self.calc_x_step(self.last_params, new_params)
        if self.calc_separated:
            x_step_converged_a = x_step_a & features_updated
            x_step_converged_b = x_step_b & features_updated_b
            epoch_step_converged_b = not_converged_b & x_step_converged_b

        else:
            x_step_converged_a = x_step_a & x_step_b & features_updated
        epoch_step_converged_a = not_converged_a & x_step_converged_a
        # print('x_step: ', x_step_converged_a[0], x_step_converged_b[0])

        # In case we use irls_tr/irls_gd_tr or nr_tr, we can also utilize the trusted region radius.
        # For now it must not be below the threshold for the X step of the loc model.
        if hasattr(optimizer_object, 'trusted_region_mode') \
                and optimizer_object.trusted_region_mode:
            converged_tr = optimizer_object.tr_radius.numpy() < pkg_constants.TRTOL_BY_FEATURE_LOC
            if hasattr(optimizer_object, 'tr_radius_b') and self.estimator._train_scale:
                converged_tr_b = \
                    optimizer_object.tr_radius_b.numpy() < pkg_constants.TRTOL_BY_FEATURE_SCALE
                epoch_tr_converged_b = not_converged_b & converged_tr_b
                epoch_step_converged_b |= epoch_tr_converged_b
            epoch_tr_converged = not_converged_a & converged_tr
            epoch_step_converged_a |= epoch_tr_converged
        # print('tr: ', epoch_tr_converged[0], epoch_tr_converged_b[0])
        # print(self.estimator.model.model_vars.converged[0], self.estimator.model.model_vars.updated[0])
        # print(self.estimator.model.model_vars.converged_b[0], self.estimator.model.model_vars.updated_b[0])
        ###########################################################
        # FINAL PART: exchange the current with the new containers.
        ####
        self.previous_number_converged = np.sum(self.estimator.model.model_vars.total_converged)
        self.last_params = new_params
        self.last_ll = new_ll
        converged_a = np.logical_or.reduce((converged_a, epoch_ll_converged_a, epoch_grad_converged_a, epoch_step_converged_a))
        if self.calc_separated:
            converged_b = np.logical_or.reduce((converged_b, epoch_ll_converged_b, epoch_grad_converged_b, epoch_step_converged_b))
            self.estimator.model.model_vars.total_converged = converged_a & converged_b
            self.estimator.model.model_vars.converged_b = converged_b
            epoch_ll_converged_a |= epoch_ll_converged_b
            epoch_grad_converged_a |= epoch_grad_converged_b
            epoch_step_converged_a |= epoch_step_converged_b
        else:
            self.estimator.model.model_vars.total_converged = converged_a
        self.estimator.model.model_vars.converged = converged_a
        # print(self.estimator.model.model_vars.total_converged[0])
        return epoch_ll_converged_a, epoch_grad_converged_a, epoch_step_converged_a

    def calc_x_step(self, prev_params, curr_params):
        """Calculates convergence based on the difference in parameters before and
        after the update."""
        def get_norm_converged(model: str, prev_params):
            if model == 'loc':
                idx_train = self.estimator.model.model_vars.idx_train_loc
                xtol = pkg_constants.XTOL_BY_FEATURE_LOC
            elif model == 'scale':
                idx_train = self.estimator.model.model_vars.idx_train_scale
                xtol = pkg_constants.XTOL_BY_FEATURE_SCALE
            else:
                assert False, "Supply either 'loc' or 'scale'!"
            x_step = curr_params - prev_params
            x_norm = np.sqrt(np.sum(np.square(x_step[idx_train, :]), axis=0))
            # print('x_norm: ', x_norm[0])
            return x_norm < xtol

        # We use a trick here: First we set both the loc and scale convergence to True.
        # It is not necessary to use an array with length = number of features, since bitwise
        # AND also works with a single boolean.
        loc_conv = np.bool_(True)
        scale_conv = np.bool_(True)

        # Now we check which models need to be trained. E.g. if you are using quick_scale = True,
        # self._train_scale will be False and so the above single True value will be used.
        if self.estimator._train_loc:
            loc_conv = get_norm_converged('loc', prev_params)
        if self.estimator._train_scale:
            scale_conv = get_norm_converged('scale', prev_params)

        # Finally, we check that only features updated in this epoch can evaluate to True.
        # This is only a problem for 2nd order optims with trusted region mode, since it might
        # occur, that a feature isn't updated, so the x_step is zero although not yet converged.
        return loc_conv, scale_conv

    def getLoss(self):
        return np.sum(self.last_ll)

    def getPreviousNumberConverged(self):
        return self.previous_number_converged
