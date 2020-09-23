from typing import TypeVar, Union

import scipy
import dask
import numpy as np

try:
    from anndata import AnnData
except ImportError:
    AnnData = TypeVar("AnnData")

ArrayLike = Union[np.ndarray, scipy.sparse.csr_matrix, dask.array.core.Array]
InputType = Union[ArrayLike, AnnData, "InputDataBase"]