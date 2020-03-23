import numpy as np
import tensorflow as tf
from .external import pkg_constants


class ConvergenceCalculator:
    """Wrapper object containing all necessary methods to calculate convergences based on change
    in likelihood, gradient and parameters."""

    def __init__(self, estimator, last_ll: np.ndarray):
        self.estimator = estimator
        self.current_converged = estimator.model.model_vars.converged
        self.current_params = estimator.model.params_copy
        self.current_ll = last_ll
        self.previous_number_converged = 0

    def calculate_convergence(self, results, jac_normalization, optimizer_object, batch_features):
        """Calculates convergence based on change in likelihood, gradient and parameters."""

        features_updated = self.estimator.model.model_vars.features_updated
        total_converged = self.estimator.model.model_vars.converged
        not_converged_prev = ~ self.current_converged
        n_features = self.estimator.input_data.n_features

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
            indices = np.where(not_converged_prev)[0]
            updated_lls = tf.scatter_nd(
                np.expand_dims(indices, 1), new_ll, shape=[n_features])
            # fill the added columns with previous ll
            new_ll = np.where(not_converged_prev, updated_lls.numpy(), self.current_ll)

            # fill added columns with the gradients from previous runs.
            indices = tf.where(not_converged_prev)
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
                ).numpy()
            )
            # TODO: added columns are zero here, does that matter?
        else:
            new_ll = new_ll.numpy()

        ###########################################################
        # SECOND PART: Calculate ll convergence.
        ####

        # Get all converged features due to change in ll < LLTOL_BY_FEATURE
        # IMPORTANT: we need to ensure they have also been updated, otherwise ll_prev = ll_current!
        ll_difference = np.abs(self.current_ll - new_ll) / self.current_ll
        ll_converged = (ll_difference < pkg_constants.LLTOL_BY_FEATURE) & features_updated
        epoch_ll_converged = not_converged_prev & ll_converged  # formerly known as converged_f

        total_converged |= epoch_ll_converged

        ###########################################################
        # THIRD PART: calculate grad convergence.
        ####
        grad_loc = np.sum(grad_numpy[:, self.estimator.model.model_vars.idx_train_loc], axis=1)
        grad_norm_loc = grad_loc / jac_normalization
        grad_scale = np.sum(grad_numpy[:, self.estimator.model.model_vars.idx_train_scale], axis=1)
        grad_norm_scale = grad_scale / jac_normalization

        grad_norm_loc_converged = grad_norm_loc < pkg_constants.GTOL_BY_FEATURE_LOC
        grad_norm_scale_converged = grad_norm_scale < pkg_constants.GTOL_BY_FEATURE_SCALE

        grad_converged = grad_norm_loc_converged & grad_norm_scale_converged & features_updated
        epoch_grad_converged = not_converged_prev & grad_converged  # formerly known as converged_g

        total_converged |= grad_converged

        ###########################################################
        # Fourth PART: calculate parameter step convergence.
        ####
        x_step_converged = self.calc_x_step(self.current_params, new_params, features_updated)
        epoch_step_converged = not_converged_prev & x_step_converged

        # In case we use irls_tr/irls_gd_tr or nr_tr, we can also utilize the trusted region radius.
        # For now it must not be below the threshold for the X step of the loc model.
        if hasattr(optimizer_object, 'trusted_region_mode') \
                and optimizer_object.trusted_region_mode:
            converged_tr = optimizer_object.tr_radius.numpy() < pkg_constants.TRTOL_BY_FEATURE_LOC
            if hasattr(optimizer_object, 'tr_radius_b') and self.estimator.train_scale:
                converged_tr &= \
                    optimizer_object.tr_radius_b.numpy() < pkg_constants.TRTOL_BY_FEATURE_SCALE
            epoch_tr_converged = not_converged_prev & converged_tr
            epoch_step_converged |= epoch_tr_converged

        total_converged |= epoch_step_converged

        ###########################################################
        # FINAL PART: exchange the current with the new containers.
        ####
        self.previous_number_converged = np.sum(self.current_converged)
        self.current_converged = total_converged.copy()
        self.current_params = new_params
        self.current_ll = new_ll

        return total_converged, epoch_ll_converged, epoch_grad_converged, epoch_step_converged

    def calc_x_step(self, prev_params, curr_params, features_updated):
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
            return x_norm < xtol

        # We use a trick here: First we set both the loc and scale convergence to True.
        # It is not necessary to use an array with length = number of features, since bitwise
        # AND also works with a single boolean.
        loc_conv = np.bool_(True)
        scale_conv = np.bool_(True)

        # Now we check which models need to be trained. E.g. if you are using quick_scale = True,
        # self._train_scale will be False and so the above single True value will be used.
        if self.estimator.train_loc:
            loc_conv = get_norm_converged('loc', prev_params)
        if self.estimator.train_scale:
            scale_conv = get_norm_converged('scale', prev_params)

        # Finally, we check that only features updated in this epoch can evaluate to True.
        # This is only a problem for 2nd order optims with trusted region mode, since it might
        # occur, that a feature isn't updated, so the x_step is zero although not yet converged.
        return loc_conv & scale_conv & features_updated

    def getLoss(self):
        return np.sum(self.current_ll)

    def getNumberConverged(self):
        return np.sum(self.current_converged)

    def getPreviousNumberConverged(self):
        return self.previous_number_converged
