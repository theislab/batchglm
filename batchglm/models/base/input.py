import dask.array
import logging
import numpy as np
import scipy.sparse
import sparse
from typing import List

try:
    import anndata
    try:
        from anndata.base import Raw
    except ImportError:
        from anndata import Raw
except ImportError:
    anndata = None
    Raw = None

logger = logging.getLogger(__name__)


class InputDataBase:
    """
    Base class for all input data types.
    """
    features: List[str]
    observations: List[str]
    chunk_size_cells: int
    chunk_size_genes: int

    def __init__(
            self,
            data,
            observation_names=None,
            feature_names=None,
            chunk_size_cells: int = 100000,
            chunk_size_genes: int = 100,
            as_dask: bool = True,
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
        if isinstance(data, np.ndarray) or \
                isinstance(data, scipy.sparse.csr_matrix) or \
                isinstance(data, dask.array.core.Array):
            self.x = data
        elif isinstance(data, anndata.AnnData) or isinstance(data, Raw):
            self.x = data.X
        elif isinstance(data, InputDataBase):
            self.x = data.x
        else:
            raise ValueError("type of data %s not recognized" % type(data))

        if as_dask:
            if isinstance(self.x, dask.array.core.Array):
                self.x = self.x.compute()
            # Need to wrap dask around the COO matrix version of the sparse package if matrix is sparse.
            if isinstance(self.x, scipy.sparse.spmatrix):
                self.x = dask.array.from_array(
                    sparse.COO.from_scipy_sparse(
                        self.x.astype(cast_dtype if cast_dtype is not None else self.x.dtype)
                    ),
                    chunks=(chunk_size_cells, chunk_size_genes),
                    asarray=False
                )
            else:
                self.x = dask.array.from_array(
                    self.x.astype(cast_dtype if cast_dtype is not None else self.x.dtype),
                    chunks=(chunk_size_cells, chunk_size_genes),
                )
        else:
            if isinstance(self.x, dask.array.core.Array):
                self.x = self.x.compute()
            if cast_dtype is not None:
                self.x = self.x.astype(cast_dtype)

        self._feature_allzero = np.sum(self.x, axis=0) == 0
        self.chunk_size_cells = chunk_size_cells
        self.chunk_size_genes = chunk_size_genes

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

    def fetch_x_dense(self, idx):
        assert isinstance(self.x, np.ndarray), "tried to fetch dense from non ndarray"

        return self.x[idx, :]

    def fetch_x_sparse(self, idx):
        assert isinstance(self.x, scipy.sparse.csr_matrix), "tried to fetch sparse from non csr_matrix"

        data = self.x[idx, :]

        data_idx = np.asarray(np.vstack(data.nonzero()).T, np.int64)
        data_val = np.asarray(data.data, np.float64)
        data_shape = np.asarray(data.shape, np.int64)

        if idx.shape[0] == 1:
            data_val = np.squeeze(data_val, axis=0)
            data_idx = np.squeeze(data_idx, axis=0)

        return data_idx, data_val, data_shape
