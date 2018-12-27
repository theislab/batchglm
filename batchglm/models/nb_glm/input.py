try:
    import anndata
except ImportError:
    anndata = None

from .external import InputData_GLM
from .external import INPUT_DATA_PARAMS

INPUT_DATA_PARAMS = INPUT_DATA_PARAMS.copy()

class InputData_NBGLM(InputData_GLM):
    """
    Input data for Generalized Linear Models (GLMs) with negative binomial noise.
    """