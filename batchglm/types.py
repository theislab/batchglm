from typing import TypeAlias, TypeVar, Union

import scipy
import dask

try:
    from anndata import AnnData
except ImportError:
    AnnData = TypeVar("AnnData")

ArrayLike = TypeAlias(Union[np.ndarray, scipy.sparse.csr_matrix, dask.array.core.Array])
InputType = TypeAlias(Union[ArrayLike, AnnData, "InputDataBase"])
