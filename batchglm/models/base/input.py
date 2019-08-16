import logging
import numpy as np
import scipy
import scipy.sparse
from typing import List

try:
    import anndata
except ImportError:
    anndata = None

logger = logging.getLogger(__name__)


class _InputDataBase:
    """
    Base class for all input data types.
    """
    features: List[str]
    observations: List[str]

    def __init__(
            self,
            data,
            observation_names=None,
            feature_names=None,
            cast_dtype=None
    ):
        """
        Create a new InputData object.

        :param data: Some data object.

        Can be either:
            - np.ndarray: NumPy array containing the raw data
            - anndata.AnnData: AnnData object containing the count data and optional the design models
                stored as data.obsm[design_loc] and data.obsm[design_scale]
        :param observation_names: (optional) names of the observations.
        :param feature_names: (optional) names of the features.
        :param cast_dtype: data type of all data; should be either float32 or float64
        :return: InputData object
        """
        self.observations = observation_names
        self.features = feature_names
        if isinstance(data, np.ndarray):
            self.x = data
        elif isinstance(data, anndata.AnnData) or isinstance(data, anndata.Raw):
            self.x = data.X

        if cast_dtype is not None:
            self.x = self.x.astype(cast_dtype)

        self._feature_allzero = np.sum(self.x, axis=0) == 0

    @property
    def num_observations(self):
        return self.x.shape[0]

    @property
    def num_features(self):
        return self.x.shape[1]

    @property
    def feature_isnonzero(self):
        return ~self._feature_allzero

    @property
    def feature_isallzero(self):
        return self._feature_allzero

    def fetch_X_dense(self, idx):
        return self.x[idx].values

    def fetch_X_sparse(self, idx):
        assert isinstance(self.x.X, scipy.sparse.csr_matrix), "tried to fetch sparse from non csr matrix"

        data = self.x[idx]

        data_idx = np.asarray(np.vstack(data.nonzero()).T, np.int64)
        data_val = np.asarray(data.data, np.float64)
        data_shape = np.asarray(data.shape, np.int64)

        if idx.shape[0] == 1:
            data_val = np.squeeze(data_val, axis=0)
            data_idx = np.squeeze(data_idx, axis=0)

        return data_idx, data_val, data_shape
