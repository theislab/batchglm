try:
    import anndata
except ImportError:
    anndata = None

import numpy as np
import pandas as pd
import patsy
import scipy.sparse
from typing import Union

from .utils import parse_constraints, parse_design
from .external import InputDataBase


class InputDataGLM(InputDataBase):
    """
    Input data for Generalized Linear Models (GLMs).
    """
    loc_names: list
    design_loc_names: list
    scale_names: list
    design_scale_names: list

    def __init__(
            self,
            data: Union[np.ndarray, anndata.AnnData, scipy.sparse.csr_matrix],
            design_loc: Union[np.ndarray, pd.DataFrame, patsy.design_info.DesignMatrix] = None,
            design_loc_names: Union[list, np.ndarray] = None,
            design_scale: Union[np.ndarray, pd.DataFrame, patsy.design_info.DesignMatrix] = None,
            design_scale_names: Union[list, np.ndarray] = None,
            constraints_loc: Union[np.ndarray] = None,
            constraints_scale: Union[np.ndarray] = None,
            size_factors=None,
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
        :param design_loc: Some matrix format (observations x mean model parameters)
            The location design model. Optional if already specified in `data`
        :param design_loc_names: (optional)
            Names of the design_loc parameters.
            The names might already be included in `design_loc`.
            Will be used to find identical columns in two models.
        :param design_scale: Some matrix format (observations x dispersion model parameters)
            The scale design model. Optional if already specified in `data`
        :param design_scale_names: (optional)
            Names of the design_scale parameters.
            The names might already be included in `design_loc`.
            Will be used to find identical columns in two models.
        :param constraints_loc: tensor (all parameters x dependent parameters)
            Tensor that encodes how complete parameter set which includes dependent
            parameters arises from indepedent parameters: all = <constraints, indep>.
            This tensor describes this relation for the mean model.
            This form of constraints is used in vector generalized linear models (VGLMs).
        :param constraints_scale: tensor (all parameters x dependent parameters)
            Tensor that encodes how complete parameter set which includes dependent
            parameters arises from indepedent parameters: all = <constraints, indep>.
            This tensor describes this relation for the dispersion model.
            This form of constraints is used in vector generalized linear models (VGLMs).
        :param size_factors: np.ndarray (observations)
            Constant scale factors of the mean model in the linker space.
        :param observation_names: (optional)
            Names of the observations.
        :param feature_names: (optional)
            Names of the features.
        :param cast_dtype:
            If this option is set, all provided data will be casted to this data type.
        :return: InputData object
        """
        InputDataBase.__init__(
            self=self,
            data=data,
            observation_names=observation_names,
            feature_names=feature_names,
            cast_dtype=cast_dtype
        )

        design_loc, design_loc_names = parse_design(
            design_matrix=design_loc,
            param_names=design_loc_names
        )
        design_scale, design_scale_names = parse_design(
            design_matrix=design_scale,
            param_names=design_scale_names
        )
        if cast_dtype is not None:
            design_loc = design_loc.astype(cast_dtype)
            design_scale = design_scale.astype(cast_dtype)

        self.design_loc = design_loc
        self.design_scale = design_scale
        self._design_loc_names = design_loc_names
        self._design_scale_names = design_scale_names

        constraints_loc, loc_names = parse_constraints(
            dmat=design_loc,
            dmat_par_names=design_loc_names,
            constraints=constraints_loc,
            constraint_par_names=None
        )
        constraints_scale, scale_names = parse_constraints(
            dmat=design_scale,
            dmat_par_names=design_scale_names,
            constraints=constraints_scale,
            constraint_par_names=None
        )
        self.constraints_loc = constraints_loc
        self.constraints_scale = constraints_scale
        self._loc_names = loc_names
        self._scale_names = scale_names

        self.size_factors = size_factors

    @property
    def design_loc_names(self):
        return self._design_loc_names

    @property
    def design_scale_names(self):
        return self._design_scale_names

    @property
    def loc_names(self):
        return self._loc_names

    @property
    def scale_names(self):
        return self._scale_names

    @property
    def num_design_loc_params(self):
        return self.design_loc.shape[1]

    @property
    def num_design_scale_params(self):
        return self.design_scale.shape[1]

    @property
    def num_loc_params(self):
        return self.constraints_loc.shape[1]

    @property
    def num_scale_params(self):
        return self.constraints_scale.shape[1]

    def fetch_design_loc(self, idx):
        return self.design_loc[idx, :]

    def fetch_design_scale(self, idx):
        return self.design_scale[idx, :]

    def fetch_size_factors(self, idx):
        return self.size_factors[idx]
