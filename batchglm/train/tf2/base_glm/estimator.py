import abc
import sys
import logging
import time
import numpy as np
from .generator import DataGenerator
from .convergence import ConvergenceCalculator
import tensorflow as tf
from .model import GLM
from .external import TFEstimator, _EstimatorGLM
from .optim import NR, IRLS
from .external import pkg_constants

logger = logging.getLogger("batchglm")


class Estimator(TFEstimator, _EstimatorGLM, metaclass=abc.ABCMeta):
    """
    Estimator for Generalized Linear Models (GLMs).
    """
    model: GLM
    _train_loc: bool
    _train_scale: bool
    _initialized: bool = False
    noise_model: str
    irls_algo: bool = False
    nr_algo: bool = False
    optimizer = None

    def initialize(self, **kwargs):
        self.values = []
        self.times = []
        self.converged = []
        self.lls = []
        self._initialized = True
        self.model = None

    def update(self, results, *args):
        self.optimizer.apply_gradients([(results[1], self.model.params_copy)])
        self.model.model_vars.updated = ~self.model.model_vars.converged

    def finalize(self, **kwargs):
        """
        Evaluate all tensors that need to be exported from session,
        save these as class attributes and close session.
        Changes .model entry from tf-based EstimatorGraph to numpy based Model instance and
        transfers relevant attributes.
        """
        a_var, b_var = self.model.unpack_params(
            [self.model.params, self.model.model_vars.a_var.get_shape()[0]])
        self.model = self.get_model_container(self.input_data)
        self.model._a_var = a_var.numpy()
        self.model._b_var = b_var.numpy()
        self._loss = tf.reduce_sum(
            tf.negative(self._log_likelihood) / self.input_data.num_observations).numpy()

    def __init__(
            self,
            input_data,
            dtype,
    ):
        TFEstimator.__init__(self=self, input_data=input_data, dtype=dtype)
        _EstimatorGLM.__init__(self=self, model=None, input_data=input_data)

    def _train(
            self,
            noise_model: str,
            is_batched: bool = False,
            batch_size: int = 5000,
            optimizer_object: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(),
            convergence_criteria: str = "step",
            stopping_criteria: int = 1000,
            autograd: bool = False,
            featurewise: bool = True,
            benchmark: bool = False,
            optim_algo: str = "adam",
            b_update_freq = 1
    ):
        # define some useful shortcuts here
        n_obs = self.input_data.num_observations
        n_features = self.input_data.num_features
        # set necessary attributes
        self.noise_model = noise_model
        optim = optim_algo.lower()
        self.irls_algo = optim.startswith('irls')
        self.nr_algo = optim in ['nr', 'nr_tr']

        ################################################
        # INIT Step 1: Consistency Checks
        ####
        assert not is_batched, "The TF2 backend does not yet support updates on individual" \
            "batches. Use full data updates instead."
        assert convergence_criteria in ["step", "all_converged"], \
            ("Unrecognized convergence criteria %s", convergence_criteria)

        if not self._initialized:
            raise RuntimeError("Cannot train the model: Estimator not initialized. \
                Did you forget to call estimator.initialize() ?")

        if b_update_freq == 0:
            b_update_freq = 1

        if autograd and optim_algo.lower() in ['nr', 'nr_tr']:
            logger.warning(
                "Automatic differentiation is currently not supported for hessians. Falling back \
                to closed form. Only Jacobians are calculated using autograd.")

        if featurewise and not (self.irls_algo or self.nr_algo):
            featurewise = False
            logger.warning("WARNING: 'Featurewise batching' is only available for 2nd order "
                           "optimizers IRLS and NR. Fallback to full featurespace fitting.")
        if batch_size > n_obs:
            batch_size = n_obs

        ################################################
        # INIT Step 2: Intialise training loop.
        #

        # create a tensorflow dataset using the DataGenerator
        datagenerator = DataGenerator(self, noise_model, is_batched, batch_size)
        epoch_set = datagenerator.new_epoch_set()

        # first model call to initialise prior to first update.
        epochs_until_b_update = b_update_freq - 1
        compute_b = epochs_until_b_update == 0
        for i, x_batch in enumerate(epoch_set):
            if i == 0:
                results = self.model(x_batch, compute_b=compute_b)
            else:
                results = [tf.math.add(results[i], x) for i, x in enumerate(self.model(x_batch, compute_b=compute_b))]

        # create ConvergenceCalculator to check for new convergences.
        conv_calc = ConvergenceCalculator(self, tf.negative(tf.divide(results[0], n_obs)).numpy())

        # termination decision for training loop
        def convergence_decision(num_converged, train_step):
            not_done_fitting = num_converged < n_features
            if convergence_criteria == "step":
                not_done_fitting &= train_step < stopping_criteria
            return not_done_fitting

        # condition variables neede during while loop
        batch_features = False
        train_step = 0
        num_converged = 0
        num_converged_prev = 0
        need_new_epoch_set = False
        n_conv_last_featurewise_batch = 0

        ################################################
        # Training Loop: Model Fitting happens here.
        ####

        while convergence_decision(num_converged, train_step):

            if benchmark:
                t0_epoch = time.time()

            ############################################
            # 1. recalculate, only done if featurewise
            if need_new_epoch_set:
                need_new_epoch_set = False
                # this is executed only if a new feature converged in the last train step and
                # using featurewise.
                epoch_set = datagenerator.new_epoch_set(batch_features=batch_features)
                if pkg_constants.FEATUREWISE_RECALCULATE:
                    for i, x_batch in enumerate(epoch_set, compute_b=compute_b):
                        results = self.model(x_batch) if i == 0 else \
                            [tf.math.add(results[i], x) for i, x in enumerate(self.model(x_batch, compute_b=compute_b))]

            ############################################
            # 2. Update the parameters
            self.update(results, epoch_set, batch_features, epochs_until_b_update == 0)
            ############################################
            # 3. calculate new ll, jacs, hessian/fim
            compute_b = epochs_until_b_update < 2
            for i, x_batch in enumerate(epoch_set):
                # need new params_copy in model in case we use featurewise without recalculation
                results = self.model(x_batch, compute_b=compute_b) if i == 0 \
                    else [tf.math.add(results[i], x) for i, x in enumerate(self.model(x_batch, compute_b=compute_b))]

            ############################################
            # 4. check for any new convergences
            convergences = conv_calc.calculate_convergence(
                results=results,
                jac_normalization=batch_size if is_batched else n_obs,
                optimizer_object=optimizer_object,
                batch_features=batch_features
            )

            num_converged = np.sum(self.model.model_vars.total_converged)
            loss = conv_calc.getLoss()
            if self.irls_algo and self._train_scale:
                num_updated = np.sum(
                    np.logical_or(self.model.model_vars.updated, self.model.model_vars.updated_b))
            else:
                num_updated = np.sum(self.model.model_vars.updated)
            log_output = f"Step: {train_step} loss: {loss}, "\
                f"converged {num_converged}, updated {num_updated}"
            num_converged_prev = conv_calc.getPreviousNumberConverged()

            ############################################
            # 5. report any new convergences
            if num_converged == num_converged_prev:
                logger.warning(log_output)
            else:
                if featurewise:
                    if not batch_features:
                        batch_features = True
                        self.model.batch_features = batch_features
                    conv_diff = num_converged - n_conv_last_featurewise_batch
                    if pkg_constants.FEATUREWISE_THRESHOLD < 1:
                        conv_diff /= n_features-n_conv_last_featurewise_batch
                    # Update params if number of new convergences since last
                    # featurewise batch is reached again.
                    if conv_diff >= pkg_constants.FEATUREWISE_THRESHOLD:
                        need_new_epoch_set = True
                        n_conv_last_featurewise_batch = num_converged
                        self.model.apply_featurewise_updates(conv_calc.last_params)
                        if not pkg_constants.FEATUREWISE_RECALCULATE:
                            results = self.mask_unconverged(results)
                        self.model.model_vars.remaining_features = \
                            ~self.model.model_vars.total_converged
                        self.model.featurewise_batch()

                sums = [np.sum(convergence_vals) for convergence_vals in convergences]
                log_output = f"{log_output} logs: {sums[0]} grad: {sums[1]}, "\
                    f"x_step: {sums[2]}"
                logger.warning(log_output)

            train_step += 1
            epochs_until_b_update = (epochs_until_b_update + b_update_freq - 1) % b_update_freq

            # make sure loc is not updated any longer if completely converged
            if b_update_freq > 1 and epochs_until_b_update > 1:
                if np.all(self.model.model_vars.converged):
                    epochs_until_b_update = 1  # must not be 0: scale grads not yet calculated
                    b_update_freq = 1  # from now on, calc scale grads in each step

            # store some useful stuff for benchmarking purposes.
            if benchmark:
                t1_epoch = time.time()
                self.times.append(t1_epoch-t0_epoch)
                self.converged.append(num_converged)
                self.values.append(self.model.trainable_variables[0].numpy().copy())
                self.lls.append(conv_calc.last_ll)

        ################################################
        # Final model call on the full feature space.
        ####
        logger.warning("Final Evaluation run.")
        if batch_features:
            # need to update `model.params` if conv_diff wasn't reached in last train step
            # as updates since the last featurewise batch are not yet applied in that case.
            if np.any(self.model.model_vars.remaining_features):
                self.model.apply_featurewise_updates(conv_calc.last_params)
            # now make sure we use the full feature space for the last update
            self.model.model_vars.remaining_features = np.ones(n_features, dtype=np.bool)
            self.model.featurewise_batch()

        batch_features = False  # reset in case train is run repeatedly
        # change to hessian mode since we still use hessian instead of FIM for self._fisher_inv
        self.model.setMethod('nr_tr')  # TODO: maybe stay with irls to compute fim in the future
        self.model.hessian.compute_b = True  # since self._train_scale could be False.

        # need new set here with full feature space
        # TODO: only needed if batch_features, maybe put this in the above if switch later
        final_set = datagenerator.new_epoch_set()
        for i, x_batch in enumerate(final_set):
            results = self.model(x_batch) if i == 0 else \
                [tf.math.add(results[i], x) for i, x in enumerate(self.model(x_batch))]

        # store all the final results in this estimator instance.
        self._log_likelihood = results[0].numpy()
        self._jacobian = tf.reduce_sum(tf.abs(results[1] / n_obs), axis=1)
        self._hessian = - results[2].numpy()

        fisher_inv = np.zeros_like(self._hessian)
        invertible = np.where(np.linalg.cond(self._hessian, p=None) < 1 / sys.float_info.epsilon)[0]
        num_non_invertible = n_features - len(invertible)
        if num_non_invertible > 0:
            logger.warning(f"fisher_inv could not be calculated for {num_non_invertible} features!")
        fisher_inv[invertible] = np.linalg.inv(-self._hessian[invertible])
        self._fisher_inv = fisher_inv.copy()
        self.model.hessian.compute_b = self.model.compute_b  # reset if not self._train_scale

    def update_params(self, batches, results, batch_features, update_func):
        """Wrapper method to conduct updates based on different optimizers/conditions."""
        if self.irls_algo or self.nr_algo:
            if self.irls_algo:
                # separate loc and scale update if using IRLS.
                update_func(
                    inputs=[batches, *results],
                    compute_a=True,
                    compute_b=False,
                    batch_features=batch_features,
                    is_batched=False
                )
                if self._train_scale:
                    update_func(
                        inputs=[batches, *results],
                        compute_a=False,
                        compute_b=True,
                        batch_features=batch_features,
                        is_batched=False
                    )
            else:
                update_func(
                    inputs=[batches, *results],
                    batch_features=batch_features,
                    is_batched=False
                )
        else:
            update_var = results[1]
            update_func([(update_var, self.model.params_copy)])
            self.model.model_vars.updated = ~self.model.model_vars.converged

    def mask_unconverged(self, results):

        # the idx from unconverged features, thus features reamining in the curent results
        idx = np.where(self.model.model_vars.remaining_features)[0]
        # the new remaining_features in reduced feature space
        mask = ~(self.model.model_vars.total_converged[idx])

        ll = tf.boolean_mask(results[0], mask)
        if self.irls_algo:
            jac_a = tf.boolean_mask(results[1], mask)
            jac_b = tf.boolean_mask(results[2], mask)
            fim_a = tf.boolean_mask(results[3], mask)
            fim_b = tf.boolean_mask(results[4], mask)
            return ll, jac_a, jac_b, fim_a, fim_b
        elif self.nr_algo:
            jac = tf.boolean_mask(results[1], mask)
            hessian = tf.boolean_mask(results[2], mask)
            return ll, jac, hessian
        else:
            jac = tf.boolean_mask(results[1], mask, axis=1)
        return ll, jac

    def get_optimizer_object(self, optimizer: str, learning_rate):
        """ Creates an optimizer object based on the given optimizer string."""
        optimizer = optimizer.lower()
        if optimizer == "gd":
            optim_obj = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer == "adam":
            optim_obj = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == "adagrad":
            optim_obj = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        elif optimizer == "rmsprop":
            optim_obj = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            tr_mode = optimizer.endswith('tr')
            init_dict = {
                "dtype": self.dtype,
                "model": self.model,
                "name": optimizer,
                "tr_mode": tr_mode,
                "n_obs": self.input_data.num_observations
            }
            if optimizer.startswith('irls'):
                optim_obj = IRLS(**init_dict)
            elif optimizer.startswith('nr'):
                optim_obj = NR(**init_dict)
            else:
                optim_obj = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                logger.warning("No valid optimizer given. Default optimizer Adam chosen.")

        return optim_obj

    @staticmethod
    def get_init_from_model(init_a, init_b, input_data, init_model):
        # Locations model:
        if isinstance(init_a, str) and (init_a.lower() == "auto" or init_a.lower() == "init_model"):
            my_loc_names = set(input_data.loc_names)
            my_loc_names = my_loc_names.intersection(set(init_model.input_data.loc_names))

            init_loc = np.zeros([input_data.num_loc_params, input_data.num_features])
            for parm in my_loc_names:
                init_idx = np.where(init_model.input_data.loc_names == parm)[0]
                my_idx = np.where(input_data.loc_names == parm)[0]
                init_loc[my_idx] = init_model.a_var[init_idx]

            init_a = init_loc

        # Scale model:
        if isinstance(init_b, str) and (init_b.lower() == "auto" or init_b.lower() == "init_model"):
            my_scale_names = set(input_data.scale_names)
            my_scale_names = my_scale_names.intersection(init_model.input_data.scale_names)

            init_scale = np.zeros([input_data.num_scale_params, input_data.num_features])
            for parm in my_scale_names:
                init_idx = np.where(init_model.input_data.scale_names == parm)[0]
                my_idx = np.where(input_data.scale_names == parm)[0]
                init_scale[my_idx] = init_model.b_var[init_idx]

            init_b = init_scale

        return init_a, init_b

    @abc.abstractmethod
    def get_model_container(self, input_data):
        pass
