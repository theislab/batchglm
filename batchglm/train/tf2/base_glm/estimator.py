import abc
import logging
import numpy as np
import scipy
import tensorflow as tf
from .model import GLM
from .external import TFEstimator, _EstimatorGLM
from .optim import NR, IRLS
from .external import pkg_constants
import time

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

    def initialize(self, **kwargs):
        self.values = []
        self.times = []
        self.converged = []
        self._initialized = True
        self.model = None

    def finalize(self, **kwargs):
        """
        Evaluate all tensors that need to be exported from session and save these as class attributes
        and close session.

        Changes .model entry from tf-based EstimatorGraph to numpy based Model instance and
        transfers relevant attributes.
        """

        a_var, b_var = self.model.unpack_params([self.model.params, self.model.model_vars.a_var.get_shape()[0]])
        self.model = self.get_model_container(self._input_data)
        self.model._a_var = a_var.numpy()
        self.model._b_var = b_var.numpy()
        self._loss = tf.reduce_sum(np.negative(self._log_likelihood) / self.input_data.num_observations).numpy()

    def __init__(
            self,
            input_data,
            dtype,
    ):

        self._input_data = input_data

        TFEstimator.__init__(
            self=self,
            input_data=input_data,
            dtype=dtype,
        )
        _EstimatorGLM.__init__(
            self=self,
            model=None,
            input_data=input_data
        )

    def _train(
            self,
            noise_model: str,
            is_batched: bool = True,
            batch_size: int = 100,
            optimizer_object: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(),
            convergence_criteria: str = "step",
            stopping_criteria: int = 1000,
            autograd: bool = False,
            featurewise: bool = True,
            benchmark: bool = False,
            optim_algo: str = "adam"
    ):
        if batch_size > self.input_data.num_observations:
            batch_size = self.input_data.num_observations
        if not self._initialized:
            raise RuntimeError("Cannot train the model: \
                                Estimator not initialized. Did you forget to call estimator.initialize() ?")

        if autograd and optim_algo.lower() in ['nr', 'nr_tr']:
            logger.warning("Automatic differentiation is currently not supported for hessians. \
                            Falling back to closed form. Only Jacobians are calculated using autograd.")

        self.noise_model = noise_model
        # Slice data and create batches

        data_ids = tf.data.Dataset.from_tensor_slices(
            (tf.range(self._input_data.num_observations, name="sample_index", dtype=tf.dtypes.int64))
        )
        if is_batched:
            data = data_ids.shuffle(buffer_size=2 * batch_size).repeat().batch(batch_size)
        else:
            data = data_ids.shuffle(buffer_size=2 * batch_size).batch(batch_size, drop_remainder=True)
        input_list = data.map(self.fetch_fn, num_parallel_calls=pkg_constants.TF_NUM_THREADS)

        # Iterate until conditions are fulfilled.
        train_step = 0

        # Set all to convergence status = False, this is needed if multiple
        # training strategies are run:
        converged_current = np.repeat(
            False, repeats=self.model.model_vars.n_features)

        def convergence_decision(convergence_status, train_step):
            if convergence_criteria == "step":
                return train_step < stopping_criteria
            elif convergence_criteria == "all_converged":
                return np.any(np.logical_not(convergence_status))
            elif convergence_criteria == "both":
                return np.any(np.logical_not(convergence_status)) and train_step < stopping_criteria
            else:
                raise ValueError("convergence_criteria %s not recognized." % convergence_criteria)

        # fill with highest possible number:
        ll_current = np.zeros([self._input_data.num_features], self.dtype) + np.nextafter(np.inf, 0, dtype=self.dtype)

        dataset_iterator = iter(input_list)
        irls_algo = False
        nr_algo = False
        if optim_algo.lower() in ['nr','nr_tr']:
            nr_algo = True
            update_func = optimizer_object.perform_parameter_update

        elif optim_algo.lower() in ['irls','irls_tr','irls_gd','irls_gd_tr']:
            irls_algo = True
            update_func = optimizer_object.perform_parameter_update

        else:
            update_func = optimizer_object.apply_gradients
        n_obs = self._input_data.num_observations

        curr_norm_loc = np.sqrt(np.sum(np.square(
            np.abs(self.model.params.numpy()[self.model.model_vars.idx_train_loc, :])), axis=0))
        curr_norm_scale = np.sqrt(np.sum(np.square(
            np.abs(self.model.params.numpy()[self.model.model_vars.idx_train_scale, :])), axis=0))

        batch_features = False
        while convergence_decision(converged_current, train_step):
            # ### Iterate over the batches of the dataset.
            # x_batch is a tuple (idx, (X_tensor, design_loc_tensor, design_scale_tensor, size_factors_tensor))
            if benchmark:
                t0_epoch = time.time()

            not_converged = np.logical_not(self.model.model_vars.converged)
            ll_prev = ll_current.copy()
            #if train_step % 10 == 0:
            logger.warning('step %i: loss: %s', train_step, np.array2string(ll_current[0:10]))

            if not is_batched:
                results = None
                x_batch = None
                first_batch = True
                for x_batch_tuple in input_list:
                    x_batch = self.getModelInput(x_batch_tuple, batch_features, not_converged)
                    current_results = self.model(x_batch)
                    if first_batch:
                        results = list(current_results)
                        first_batch = False
                    else:
                        for i, x in enumerate(current_results):
                            results[i] += x

            else:
                x_batch_tuple = next(dataset_iterator)
                x_batch = self.getModelInput(x_batch_tuple, batch_features, not_converged)
                results = self.model(x_batch)

            if irls_algo or nr_algo:
                if irls_algo:
                    update_func(
                        [x_batch, *results, False, n_obs],
                        True,
                        False,
                        batch_features,
                        ll_prev
                    )
                    if self._train_scale:
                        update_func(
                            [x_batch, *results, False, n_obs],
                            False,
                            True,
                            batch_features,
                            ll_prev
                        )
                else:
                    update_func(
                        [x_batch, *results, False, n_obs],
                        True,
                        True,
                        batch_features,
                        ll_prev
                    )
                features_updated = self.model.model_vars.updated
            else:
                if batch_features:
                    indices = tf.where(not_converged)
                    update_var = tf.transpose(tf.scatter_nd(
                        indices,
                        tf.transpose(results[1]),
                        shape=(self.model.model_vars.n_features, results[1].get_shape()[0])
                    ))
                else:
                    update_var = results[1]
                update_func([(update_var, self.model.params)])
                features_updated = not_converged

            if benchmark:
                self.values.append(self.model.trainable_variables[0].numpy().copy())

            # Update converged status
            prev_norm_loc = curr_norm_loc.copy()
            prev_norm_scale = curr_norm_scale.copy()
            converged_prev = converged_current.copy()
            ll_current = self.loss.norm_neg_log_likelihood(results[0]).numpy()

            if batch_features:
                indices = tf.where(not_converged)
                updated_lls = tf.scatter_nd(indices, ll_current, shape=ll_prev.shape)
                ll_current = np.where(features_updated, updated_lls.numpy(), ll_prev)

            if is_batched:
                jac_normalization = batch_size
            else:
                jac_normalization = self._input_data.num_observations
            if irls_algo:
                grad_numpy = tf.abs(tf.concat((results[1], results[2]), axis=1))
            elif nr_algo:
                grad_numpy = tf.abs(results[1])
            else:
                grad_numpy = tf.abs(tf.transpose(results[1]))
            if batch_features:
                indices = tf.where(not_converged)
                grad_numpy = tf.scatter_nd(
                    indices,
                    grad_numpy,
                    shape=(self.model.model_vars.n_features, self.model.params.get_shape()[0])
                )
            grad_numpy = grad_numpy.numpy()
            convergences = self.calculate_convergence(
                converged_prev,
                ll_prev,
                prev_norm_loc,
                prev_norm_scale,
                ll_current,
                jac_normalization,
                grad_numpy,
                features_updated,
                optimizer_object
            )
            #converged_current, converged_f, converged_g, converged_x = convergences
            converged_current = convergences[0]
            self.model.model_vars.convergence_update(converged_current, features_updated)
            num_converged = np.sum(converged_current)
            if num_converged != np.sum(converged_prev):
                if featurewise and not batch_features:
                    batch_features = True
                    self.model.batch_features = batch_features
                logger_pattern = "Step: %i loss: %f, converged %i, updated %i, (logs: %i, grad: %i, x_step: %i"
                logger_data = [
                    train_step,
                    np.sum(ll_current),
                    num_converged.astype("int32"),
                    np.sum(features_updated).astype("int32"),
                    *[np.sum(convergence_vals) for convergence_vals in convergences[1:]]
                ]
                if (irls_algo or nr_algo) and optimizer_object.trusted_region_mode:
                    logger_pattern += " tr: %i)"
                else:
                    logger_pattern += ")"

                logger.warning(
                    logger_pattern,
                    *logger_data
                )
            train_step += 1
            if benchmark:
                t1_epoch = time.time()
                self.times.append(t1_epoch-t0_epoch)
                self.converged.append(num_converged)

        # Evaluate final params
        logger.warning("Final Evaluation run.")
        self._log_likelihood = results[0].numpy()
        self._fisher_inv = tf.zeros(shape=()).numpy()
        self._hessian = tf.zeros(shape=()).numpy()
        self.model.batch_features = False

        # change to hessian mode since we still use hessian instead of FIM for self._fisher_inv
        if irls_algo:
            self.model.calc_fim = False
            self.model.calc_hessian = True
            self.model.concat_grads = True

        first_batch = True
        for x_batch_tuple in input_list:
            current_results = self.model(x_batch_tuple)
            if first_batch:
                results = list(current_results)
                first_batch = False
            else:
                for i, x in enumerate(current_results):
                    results[i] += x

        if nr_algo:
            self._hessian = results[2].numpy()
            self._jacobian = results[1].numpy()
        elif irls_algo:
            # TODO: maybe report fisher inv here. But concatenation only works if !intercept_scale
            self._fisher_inv = results[2].numpy()
            self._jacobian = results[1].numpy()

            # change back to FIM mode
            self.model.calc_fim = True
            self.model.calc_hessian = False
            self.model.concat_grads = False
        else:
            self._jacobian = results[1].numpy()

        self.model.batch_features = batch_features

    def getModelInput(self, x_batch_tuple: tuple, batch_features: bool, not_converged):
        """
            Checks whether batch_features is true and returns a smaller x_batch tuple reduced
            in feature space. Otherwise returns the x_batch.
        """
        if batch_features:
            x_tensor, design_loc_tensor, design_scale_tensor, size_factors_tensor = x_batch_tuple
            if isinstance(self._input_data.x, scipy.sparse.csr_matrix):
                not_converged_idx = np.where(not_converged)[0]
                feature_columns = tf.sparse.split(
                    x_tensor,
                    num_split=self.model.model_vars.n_features,
                    axis=1)
                feature_columns = [feature_columns[i] for i in not_converged_idx]
                x_tensor = tf.sparse.concat(axis=1, sp_inputs=feature_columns)
                if not isinstance(x_tensor, tf.sparse.SparseTensor):
                    raise RuntimeError("x_tensor now dense!!!")
            else:
                x_tensor = tf.boolean_mask(tensor=x_tensor, mask=not_converged, axis=1)
            x_batch = (x_tensor, design_loc_tensor, design_scale_tensor, size_factors_tensor)
        else:
            x_batch = x_batch_tuple

        return x_batch

    def calculate_convergence(self, converged_prev, ll_prev, prev_norm_loc, prev_norm_scale, ll_current,
                              jac_normalization, grad_numpy, features_updated, optimizer_object):
        """
            Wrapper method to perform all necessary convergence checks.
        """

        total_converged = converged_prev.copy()
        not_converged_prev = ~ converged_prev
        """
        Get all converged features due to change in ll < LLTOL_BY_FEATURE
        IMPORTANT: we need to ensure they have also been updated, otherwise ll_prev = ll_current!
        """
        ll_difference = np.abs(ll_prev - ll_current) / ll_prev
        ll_converged = (ll_difference < pkg_constants.LLTOL_BY_FEATURE) & features_updated
        epoch_ll_converged = not_converged_prev & ll_converged  # formerly known as converged_f

        total_converged |= ll_converged

        """
        Now getting convergence based on change in gradient below threshold:
        """
        grad_loc = np.sum(grad_numpy[:, self.model.model_vars.idx_train_loc], axis=1)
        grad_norm_loc = grad_loc / jac_normalization
        grad_scale = np.sum(grad_numpy[:, self.model.model_vars.idx_train_scale], axis=1)
        grad_norm_scale = grad_scale / jac_normalization

        grad_norm_loc_converged = grad_norm_loc < pkg_constants.GTOL_BY_FEATURE_LOC
        grad_norm_scale_converged = grad_norm_scale < pkg_constants.GTOL_BY_FEATURE_SCALE

        grad_converged = grad_norm_loc_converged & grad_norm_scale_converged
        epoch_grad_converged = not_converged_prev & grad_converged  # formerly known as converged_g

        total_converged |= grad_converged

        """
        Now getting convergence based on change of coefficients below threshold:
        """
        def calc_x_step(idx_train, prev_norm):
            if len(idx_train) > 0:
                curr_norm = np.sqrt(np.sum(np.square(
                    np.abs(self.model.params.numpy()[idx_train, :])
                ), axis=0))
                return np.abs(curr_norm - prev_norm)
            else:
                return np.zeros([self.model.model_vars.n_features]) + np.nextafter(np.inf, 0, dtype=self.dtype)

        x_norm_loc = calc_x_step(self.model.model_vars.idx_train_loc, prev_norm_loc)
        x_norm_scale = calc_x_step(self.model.model_vars.idx_train_scale, prev_norm_scale)

        x_norm_loc_converged = x_norm_loc < pkg_constants.XTOL_BY_FEATURE_LOC
        x_norm_scale_converged = x_norm_scale < pkg_constants.XTOL_BY_FEATURE_SCALE

        step_converged = x_norm_loc_converged & x_norm_scale_converged
        epoch_step_converged = not_converged_prev & step_converged

        total_converged |= step_converged
        """
        In case we use irls_tr/irls_gd_tr or nr_tr, we can also utilize the trusted region radius.
        For now it must not be below the threshold for the X step of the scale model.
        """
        if hasattr(optimizer_object, 'trusted_region_mode') and optimizer_object.trusted_region_mode:
            converged_tr = optimizer_object.tr_radius.numpy() < pkg_constants.TRTOL_BY_FEATURE_LOC
            if hasattr(optimizer_object, 'tr_radius_b') and self._train_scale:
                converged_tr &= optimizer_object.tr_radius_b.numpy() < pkg_constants.TRTOL_BY_FEATURE_SCALE
            epoch_tr_converged = not_converged_prev & converged_tr
            total_converged |= epoch_tr_converged
            return total_converged, epoch_ll_converged, epoch_grad_converged, epoch_step_converged, epoch_tr_converged

        return total_converged, epoch_ll_converged, epoch_grad_converged, epoch_step_converged

    def get_optimizer_object(self, optimizer: str, learning_rate):
        """
            Creates an optimizer object based on the given optimizer string.
        """

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
                "trusted_region_mode": tr_mode
            }
            if optimizer.startswith('irls'):
                optim_obj = IRLS(**init_dict)
            elif optimizer.startswith('nr'):
                optim_obj = NR(**init_dict)
            else:
                optim_obj = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                logger.warning("No valid optimizer given. Default optimizer Adam chosen.")

        return optim_obj

    def fetch_fn(self, idx):
        """
        Documentation of tensorflow coding style in this function:
        tf.py_func defines a python function (the getters of the InputData object slots)
        as a tensorflow operation. Here, the shape of the tensor is lost and
        has to be set with set_shape. For size factors, we use explicit broadcasting
        as explained below.
        """
        # Catch dimension collapse error if idx is only one element long, ie. 0D:
        if len(idx.shape) == 0:
            idx = tf.expand_dims(idx, axis=0)

        if isinstance(self._input_data.x, scipy.sparse.csr_matrix):

            x_tensor_idx, x_tensor_val, x = tf.py_function(
                func=self._input_data.fetch_x_sparse,
                inp=[idx],
                Tout=[np.int64, np.float64, np.int64],
            )
            # Note on Tout: np.float64 for val seems to be required to avoid crashing v1.12.
            x_tensor_idx = tf.cast(x_tensor_idx, dtype=tf.int64)
            x = tf.cast(x, dtype=tf.int64)
            x_tensor_val = tf.cast(x_tensor_val, dtype=self.dtype)
            x_tensor = tf.SparseTensor(x_tensor_idx, x_tensor_val, x)
            x_tensor = tf.cast(x_tensor, dtype=self.dtype)

        else:

            x_tensor = tf.py_function(
                func=self._input_data.fetch_x_dense,
                inp=[idx],
                Tout=self._input_data.x.dtype,
            )

            x_tensor.set_shape(idx.get_shape().as_list() + [self._input_data.num_features])
            x_tensor = tf.cast(x_tensor, dtype=self.dtype)

        design_loc_tensor = tf.py_function(
            func=self._input_data.fetch_design_loc,
            inp=[idx],
            Tout=self._input_data.design_loc.dtype,
        )
        design_loc_tensor.set_shape(idx.get_shape().as_list() + [self._input_data.num_design_loc_params])
        design_loc_tensor = tf.cast(design_loc_tensor, dtype=self.dtype)

        design_scale_tensor = tf.py_function(
            func=self._input_data.fetch_design_scale,
            inp=[idx],
            Tout=self._input_data.design_scale.dtype,
        )
        design_scale_tensor.set_shape(idx.get_shape().as_list() + [self._input_data.num_design_scale_params])
        design_scale_tensor = tf.cast(design_scale_tensor, dtype=self.dtype)

        if self._input_data.size_factors is not None and self.noise_model in ["nb", "norm"]:
            size_factors_tensor = tf.py_function(
                func=self._input_data.fetch_size_factors,
                inp=[idx],
                Tout=self._input_data.size_factors.dtype,
            )
            size_factors_tensor = tf.cast(size_factors_tensor, dtype=self.dtype)

        else:
            size_factors_tensor = tf.constant(1, shape=[1, 1], dtype=self.dtype)

        # feature batching
        return x_tensor, design_loc_tensor, design_scale_tensor, size_factors_tensor

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
