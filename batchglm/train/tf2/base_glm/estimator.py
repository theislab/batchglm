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
        self.model = self.get_model_container(self.input_data)
        self.model._a_var = a_var.numpy()
        self.model._b_var = b_var.numpy()
        self._loss = tf.reduce_sum(np.negative(self._log_likelihood) / self.input_data.num_observations).numpy()

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
            is_batched: bool = True,
            batch_size: int = 1000,
            optimizer_object: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(),
            convergence_criteria: str = "step",
            stopping_criteria: int = 1000,
            autograd: bool = False,
            featurewise: bool = True,
            benchmark: bool = False,
            optim_algo: str = "adam"
    ):

        conv_all = lambda x, y: not np.all(x)
        conv_step = lambda x, y: not np.all(x) and y < stopping_criteria
        assert convergence_criteria in ["step", "all_converged"], ("Unrecognized convergence criteria %s", convergence_criteria)
        convergence_decision = conv_step if convergence_criteria == "step" else conv_all

        n_obs = self.input_data.num_observations
        n_features = self.input_data.num_features
        if batch_size > n_obs:
            batch_size = n_obs
        if not self._initialized:
            raise RuntimeError("Cannot train the model: \
                                Estimator not initialized. Did you forget to call estimator.initialize() ?")

        if autograd and optim_algo.lower() in ['nr', 'nr_tr']:
            logger.warning("Automatic differentiation is currently not supported for hessians. \
                            Falling back to closed form. Only Jacobians are calculated using autograd.")

        self.noise_model = noise_model
        sparse = isinstance(self.input_data.x, scipy.sparse.csr_matrix)
        full_model = not is_batched

        def generate():
            """
            Generator for the full model.
            We use max_obs to cut the observations with max_obs % batch_size = 0 to ensure consistent
            sizes of tensors.
            """
            fetch_size_factors = self.input_data.size_factors is not None and self.noise_model in ["nb", "norm"]

            if full_model:
                max_obs = n_obs - (n_obs % batch_size)
                obs_pool = np.arange(max_obs)
            else:
                max_obs = n_obs
                obs_pool = np.random.permutation(n_obs)

            for x in range(0, max_obs, batch_size):
                idx = obs_pool[x: x + batch_size]  # numpy automatically returns only id:id+n_obs if out of range

                x = self.input_data.fetch_x_sparse(idx) if sparse else self.input_data.fetch_x_dense(idx)
                dloc = self.input_data.fetch_design_loc(idx)
                dscale = self.input_data.fetch_design_scale(idx)
                size_factors = self.input_data.fetch_size_factors(idx) if fetch_size_factors else 1

                yield x, dloc, dscale, size_factors
            return

        dtp = self.dtype
        output_types = ((tf.int64, dtp, tf.int64), *(dtp,) * 3) if sparse else (dtp,) * 4
        dataset = tf.data.Dataset.from_generator(generator=generate, output_types=output_types)
        if sparse:
            dataset = dataset.map(lambda ivs_tuple, loc, scale, sf: (tf.SparseTensor(*ivs_tuple), loc, scale, sf))


        # Set all to convergence status = False, this is needed if multiple
        # training strategies are run:
        converged_current = np.zeros(n_features, dtype=np.bool)

        # fill with lowest possible number:
        ll_current = np.nextafter(np.inf, np.zeros(n_features), dtype=self.dtype)

        irls_algo = optim_algo.lower() in ['irls', 'irls_tr', 'irls_gd', 'irls_gd_tr']
        nr_algo = optim_algo.lower() in ['nr', 'nr_tr']

        update_func = optimizer_object.perform_parameter_update if irls_algo or nr_algo else optimizer_object.apply_gradients

        prev_params = self.model.params.numpy()

        batch_features = False
        train_step = 0
        num_batches = n_obs // batch_size

        while convergence_decision(converged_current, train_step):

            if benchmark:
                t0_epoch = time.time()

            not_converged = ~ self.model.model_vars.converged
            ll_prev = ll_current.copy()
            results = None
            for i, x_batch_tuple in enumerate(dataset):
                x_batch = self.getModelInput(x_batch_tuple, batch_features, not_converged)
                current_results = self.model(x_batch)
                if is_batched or i == 0:
                    results = current_results
                else:
                    results = [tf.math.add(results[i], x) for i, x in enumerate(current_results)]

                if is_batched or i == num_batches - 1:

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
                                shape=(n_features, results[1].get_shape()[0])
                            ))
                        else:
                            update_var = results[1]
                        update_func([(update_var, self.model.params)])
                        features_updated = not_converged

                    if benchmark:
                        self.values.append(self.model.trainable_variables[0].numpy().copy())

                    # Update converged status
                    converged_prev = converged_current.copy()
                    ll_current = self.loss.norm_neg_log_likelihood(results[0]).numpy()

                    if batch_features:
                        indices = tf.where(not_converged)
                        updated_lls = tf.scatter_nd(indices, ll_current, shape=ll_prev.shape)
                        ll_current = np.where(features_updated, updated_lls.numpy(), ll_prev)

                    if is_batched:
                        jac_normalization = batch_size
                    else:
                        jac_normalization = n_obs
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
                            shape=(n_features, self.model.params.get_shape()[0])
                        )
                    grad_numpy = grad_numpy.numpy()
                    convergences = self.calculate_convergence(
                        converged_prev,
                        ll_prev,
                        ll_current,
                        prev_params,
                        jac_normalization,
                        grad_numpy,
                        features_updated,
                        optimizer_object
                    )

                    prev_params = self.model.params.numpy()
                    #converged_current, converged_f, converged_g, converged_x = convergences
                    converged_current = convergences[0]
                    self.model.model_vars.convergence_update(converged_current, features_updated)
                    num_converged = np.sum(converged_current)
                    if num_converged != np.sum(converged_prev):
                        if featurewise and not batch_features:
                            batch_features = True
                            self.model.batch_features = batch_features
                        logger_pattern = "Step: %i loss: %f, converged %i, updated %i, (logs: %i, grad: %i, x_step: %i)"
                        logger.warning(
                            logger_pattern,
                            train_step,
                            np.sum(ll_current),
                            num_converged.astype("int32"),
                            np.sum(features_updated).astype("int32"),
                            *[np.sum(convergence_vals) for convergence_vals in convergences[1:]]
                        )
                    else:
                        logger.warning('step %i: loss: %f', train_step, np.sum(ll_current))
                    train_step += 1
                    if benchmark:
                        t1_epoch = time.time()
                        self.times.append(t1_epoch-t0_epoch)
                        self.converged.append(num_converged)

        # Evaluate final params
        logger.warning("Final Evaluation run.")
        self.model.batch_features = False

        # change to hessian mode since we still use hessian instead of FIM for self._fisher_inv
        self.model.setMethod('nr_tr')
        self.model.hessian.compute_b = True

        first_batch = True
        for x_batch_tuple in dataset:
            current_results = self.model(x_batch_tuple)
            if first_batch:
                results = list(current_results)
                first_batch = False
            else:
                for i, x in enumerate(current_results):
                    results[i] += x

        for i, x_batch_tuple in enumerate(dataset):
            current_results = self.model(x_batch_tuple)
            results = current_results if i == 0 else [tf.math.add(results[i], x) for i, x in enumerate(current_results)]

        self._log_likelihood = self.loss.norm_log_likelihood(results[0].numpy())
        self._jacobian = tf.reduce_sum(tf.abs(results[1] / self.input_data.num_observations), axis=1)

        # TODO: maybe report fisher inf here. But concatenation only works if !intercept_scale
        self._fisher_inv = tf.linalg.inv(results[2]).numpy()
        self._hessian = -results[2].numpy()

        self.model.hessian.compute_b = self.model.compute_b
        self.model.batch_features = batch_features

    def getModelInput(self, x_batch_tuple: tuple, batch_features: bool, not_converged):
        """
            Checks whether batch_features is true and returns a smaller x_batch tuple reduced
            in feature space. Otherwise returns the x_batch.
        """
        if batch_features:
            x_tensor, design_loc_tensor, design_scale_tensor, size_factors_tensor = x_batch_tuple
            if isinstance(self.input_data.x, scipy.sparse.csr_matrix):
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

    def calculate_convergence(self, converged_prev, ll_prev, ll_current, prev_params,
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

        total_converged |= epoch_ll_converged

        """
        Now getting convergence based on change in gradient below threshold:
        """
        grad_loc = np.sum(grad_numpy[:, self.model.model_vars.idx_train_loc], axis=1)
        grad_norm_loc = grad_loc / jac_normalization
        grad_scale = np.sum(grad_numpy[:, self.model.model_vars.idx_train_scale], axis=1)
        grad_norm_scale = grad_scale / jac_normalization

        grad_norm_loc_converged = grad_norm_loc < pkg_constants.GTOL_BY_FEATURE_LOC
        grad_norm_scale_converged = grad_norm_scale < pkg_constants.GTOL_BY_FEATURE_SCALE

        grad_converged = grad_norm_loc_converged & grad_norm_scale_converged & features_updated
        epoch_grad_converged = not_converged_prev & grad_converged  # formerly known as converged_g

        total_converged |= grad_converged

        """
        Now getting convergence based on change of coefficients below threshold:
        """

        x_step_converged = self.calc_x_step(prev_params, features_updated)
        epoch_step_converged = not_converged_prev & x_step_converged

        """
        In case we use irls_tr/irls_gd_tr or nr_tr, we can also utilize the trusted region radius.
        For now it must not be below the threshold for the X step of the loc model.
        """
        if hasattr(optimizer_object, 'trusted_region_mode') and optimizer_object.trusted_region_mode:
            converged_tr = optimizer_object.tr_radius.numpy() < pkg_constants.XTOL_BY_FEATURE_LOC
            if hasattr(optimizer_object, 'tr_radius_b') and self._train_scale:
                converged_tr &= optimizer_object.tr_radius_b.numpy() < pkg_constants.XTOL_BY_FEATURE_SCALE
            epoch_tr_converged = not_converged_prev & converged_tr
            epoch_step_converged |= epoch_tr_converged

        total_converged |= epoch_step_converged

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

    def calc_x_step(self, prev_params, features_updated):

        def get_norm_converged(model: str, prev_params):
            if model == 'loc':
                idx_train = self.model.model_vars.idx_train_loc
                XTOL = pkg_constants.XTOL_BY_FEATURE_LOC
            elif model == 'scale':
                idx_train = self.model.model_vars.idx_train_scale
                XTOL = pkg_constants.XTOL_BY_FEATURE_SCALE
            else:
                assert False, "Supply either 'loc' or 'scale'!"
            x_step = self.model.params.numpy() - prev_params
            x_norm = np.sqrt(np.sum(np.square(x_step[idx_train, :]), axis=0))
            return x_norm < XTOL

        """
        We use a trick here: First we set both the loc and scale convergence to True.
        It is not necessary to use an array with length = number of features, since bitwise
        AND also works with a single boolean.
        """
        loc_conv = np.bool_(True)
        scale_conv = np.bool_(True)

        """
        Now we check which models need to be trained. E.g. if you are using quick_scale = True,
        self._train_scale will be False and so the above single True value will be used.
        """
        if self._train_loc:
            loc_conv = get_norm_converged('loc', prev_params)
        if self._train_scale:
            scale_conv = get_norm_converged('scale', prev_params)

        """
        Finally, we check that only features updated in this epoch can evaluate to True.
        This is only a problem for 2nd order optims with trusted region mode, since it might occur,
        that a feature isn't updated, so the x_step is zero although not yet converged.
        """
        return loc_conv & scale_conv & features_updated
