import dask.array
import logging
import numpy as np
import scipy.sparse
import sparse
from typing import List, Union, Optional, Tuple
from .external import types as T, maybe_compute

# TODO: make this nicer
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
            data: T.InputType,
            weights: Optional[Union[T.ArrayLike, str]] = None,
            observation_names: Optional[List[str]] = None,
            feature_names: Optional[List[str]] = None,
            chunk_size_cells: int = 100000,
            chunk_size_genes: int = 100,
            as_dask: bool = True,
            cast_dtype: Optional[np.dtype] = None
    ):
        """
        Create a new InputData object.

        :param data: Some data object.

        Can be either:
            - np.ndarray: NumPy array containing the raw data
            - anndata.AnnData: AnnData object containing the count data and optional the design models
                stored as data.obsm[design_loc] and data.obsm[design_scale]
        :param weights: (optional) observation weights
        :param observation_names: (optional) names of the observations.
        :param feature_names: (optional) names of the features.
        :param cast_dtype: data type of all data; should be either float32 or float64
        :return: InputData object
        """
        self.observations = observation_names
        self.features = feature_names
        self.w = None

        if isinstance(data, np.ndarray) or \
                isinstance(data, scipy.sparse.csr_matrix) or \
                isinstance(data, dask.array.core.Array):
            self.x = data
        elif isinstance(data, anndata.AnnData) or isinstance(data, Raw):
            self.x = data.X
            if isinstance(weights, str):
                self.w = data.obs[weights].values
        elif isinstance(data, InputDataBase):
            self.x = data.x
            self.w = data.w
        else:
            raise ValueError("type of data %s not recognized" % type(data))

        self._cast_type = cast_dtype if cast_dtype is not None else self.x.dtype
        self._cast_type_float = np.float32 if not issubclass(type(self._cast_type), np.floating) else self._cast_type

        if self.w is None:
            self.w = np.ones(self.x.shape[0], dtype=self._cast_type_float)

        if scipy.sparse.issparse(self.w):
            self.w = self.w.toarray()
        if self.w.ndim == 1:
            self.w = self.w.reshape((-1, 1))

        # sanity checks
        # TODO: don't use asserts
        assert self.w.shape == (self.x.shape[0], 1), "invalid weight shape %s" % self.w.shape
        assert issubclass(self.w.dtype.type, np.floating), "invalid weight type %s" % self.w.dtype

        if self.observations is not None:
            assert len(self.observations) == self.x.shape[0]
        if self.features is not None:
            assert len(self.features) == self.x.shape[1]

        self.x = maybe_compute(self.x)
        self.w = maybe_compute(self.w)

        if as_dask:
            # Need to wrap dask around the COO matrix version of the sparse package if matrix is sparse.
            if scipy.sparse.issparse(self.x):
                self.x = dask.array.from_array(
                    sparse.COO.from_scipy_sparse(self.x.astype(self._cast_type)),
                    chunks=(chunk_size_cells, chunk_size_genes),
                    asarray=False
                )
            else:
                self.x = dask.array.from_array(
                    self.x.astype(self._cast_type),
                    chunks=(chunk_size_cells, chunk_size_genes),
                )

            self.w = dask.array.from_array(self.w.astype(self._cast_type_float), chunks=(chunk_size_cells, 1))
        else:
            # TODO: fix this
            if scipy.sparse.issparse(self.x):
                raise TypeError(f"For sparse matrices, only dask is supported.")

            self.x = self.x.astype(self._cast_type)
            self.w = self.w.astype(self._cast_type)

        self._feature_allzero = np.sum(self.x, axis=0) == 0
        self.chunk_size_cells = chunk_size_cells
        self.chunk_size_genes = chunk_size_genes

    @property
    def num_observations(self) -> int:
        return self.x.shape[0]

    @property
    def num_features(self) -> int:
        return self.x.shape[1]

    @property
    def feature_isnonzero(self) -> np.ndarray:
        return ~self._feature_allzero

    @property
    def feature_isallzero(self) -> np.ndarray:
        return self._feature_allzero

    def fetch_x_dense(self, idx: T.IndexLike) -> np.ndarray:
        assert isinstance(self.x, np.ndarray), "tried to fetch dense from non ndarray"

        return self.x[idx, :]

    def fetch_x_sparse(self, idx: T.IndexLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert isinstance(self.x, scipy.sparse.csr_matrix), "tried to fetch sparse from non csr_matrix"

        data = self.x[idx, :]

        data_idx = np.asarray(np.vstack(data.nonzero()).T, np.int64)
        data_val = np.asarray(data.data, np.float64)
        data_shape = np.asarray(data.shape, np.int64)

        if idx.shape[0] == 1:
            data_val = np.squeeze(data_val, axis=0)
            data_idx = np.squeeze(data_idx, axis=0)

        return data_idx, data_val, data_shape
