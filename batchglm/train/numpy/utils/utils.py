from typing import Union, Optional

import numpy as np
from scipy.sparse import spmatrix
import dask.array

from .external import ArrayLike


def maybe_compute(array: Optional[Union[ArrayLike]], copy: bool = False) -> Optional[Union[np.ndarray, spmatrix]]:
    if array is None:
        return None
    if isdask(array):
        return array.compute()
    return array.copy() if copy else array


def isdask(array: Optional[ArrayLike]) -> bool:
    return isinstance(array, dask.array.core.Array)
