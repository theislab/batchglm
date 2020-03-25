import numpy as np
from scipy.sparse import csr_matrix
import tensorflow as tf
import time

class DataGenerator:
    """Wrapper Object to generate an iterable TensorFlow Dataset from given input data."""

    def __init__(
            self,
            estimator,
            noise_model: str,
            is_batched_model: bool,
            batch_size: int
    ):
        self.estimator = estimator
        self.noise_model = noise_model
        self.is_batched_model = is_batched_model
        self.batch_size = batch_size
        self.sparse = isinstance(estimator.input_data.x, csr_matrix)
        self.n_obs = estimator.input_data.num_observations
        # integer ceil division with arithmetic trick: ceil(a/b)=(a+b-1)//b
        # We need this for cases where n_obs mod batch_size != 0
        self.num_batches = (self.n_obs + batch_size - 1) // batch_size
        dtp = estimator.dtype
        output_types = ((tf.int64, dtp, tf.int64), *(dtp,) * 3) if self.sparse else (dtp,) * 4
        self.dataset = tf.data.Dataset.from_generator(
            generator=self._generate, output_types=output_types)

    def _generate(self):
        """
        Generates `(counts, design_loc, design_scale, size_factors)` tuples of `self.input_data`.
        The number of observations in each such data batch is given by `self.batch size`.
        If `self.is_batched_model`, the method uses a random permutation of `input_data` each time
        it is called.
        """
        input_data = self.estimator.input_data
        fetch_size_factors = input_data.size_factors is not None \
            and self.noise_model in ["nb", "norm"]
        obs_pool = np.random.permutation(self.n_obs) \
            if self.is_batched_model else np.arange(self.n_obs)
        for start_id in range(0, self.n_obs, self.batch_size):
            # numpy ignores ids > len(obs_pool) so no out of bounds check needed here:
            idx = obs_pool[start_id: start_id + self.batch_size]
            counts = input_data.fetch_x_sparse(idx) if self.sparse \
                else input_data.fetch_x_dense(idx)
            dloc = input_data.fetch_design_loc(idx)
            dscale = input_data.fetch_design_scale(idx)
            size_factors = input_data.fetch_size_factors(idx) if fetch_size_factors else 1
            yield counts, dloc, dscale, size_factors

    def _featurewise_batch_sparse(self, ivs_tuple, dloc, dscale, size_factors):

        not_converged = ~self.estimator.model.model_vars.total_converged

        not_converged_numeric = tf.cast(not_converged, dtype=tf.int64)
        col_idx_map = tf.cumsum(not_converged_numeric, exclusive=True)
        ivs_tuple[0].set_shape([None, 2])
        mask = tf.gather(not_converged, ivs_tuple[0][:, 1])
        new_indices = tf.gather_nd(ivs_tuple[0], tf.where(mask))
        row_idx, col_idx = tf.split(new_indices, num_or_size_splits=2, axis=1, num=2)

        new_indices = tf.concat([row_idx, tf.gather(col_idx_map, col_idx)], 1)
        new_values = tf.boolean_mask(ivs_tuple[1], mask)
        n_features = col_idx_map[-1]
        if not_converged[-1]:
            n_features += 1

        x_tensor = (new_indices, new_values, (ivs_tuple[2][0], n_features))
        return x_tensor, dloc, dscale, size_factors

    def _featurewise_batch(self, x_tensor, dloc, dscale, size_factors):
        """Takes an element of a dataset, performs featurewise batching
        and returns the reduced element."""

        not_converged = ~self.estimator.model.model_vars.total_converged
        """
        if self.sparse:
            feature_columns = tf.sparse.split(
                x_tensor,
                num_split=self.estimator.model_vars.n_features,
                axis=1)
            not_converged_idx = np.where(not_converged)[0]
            feature_columns = [feature_columns[i] for i in not_converged_idx]
            x_tensor = tf.sparse.concat(axis=1, sp_inputs=feature_columns)
        else:
        """
        x_tensor = tf.boolean_mask(tensor=x_tensor, mask=not_converged, axis=1)
        return x_tensor, dloc, dscale, size_factors

    def new_epoch_set(self, batch_features: bool = False):
        """Returns an iterable TensorFlow Dataset of the input data."""
        dataset_to_return = self.dataset.take(self.num_batches)

        if self.sparse:
            if batch_features:
                dataset_to_return = dataset_to_return.map(self._featurewise_batch_sparse)
            dataset_to_return = dataset_to_return.map(
                lambda ivs_tuple, loc, scale, sf: (tf.SparseTensor(*ivs_tuple), loc, scale, sf)
            )
        else:
            if batch_features:
                dataset_to_return = dataset_to_return.map(self._featurewise_batch)

        return dataset_to_return.cache().prefetch(1)
