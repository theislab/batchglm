from typing import TypeVar, Union

import dask
import numpy as np
from scipy.sparse import spmatrix

try:
    from anndata import AnnData
except ImportError:
    AnnData = TypeVar("AnnData")

ArrayLike = Union[np.ndarray, spmatrix, dask.array.core.Array]
IndexLike = Union[np.ndarray, tuple, list, int]  # not exhaustive
InputType = Union[ArrayLike, AnnData, "InputDataBase"]
