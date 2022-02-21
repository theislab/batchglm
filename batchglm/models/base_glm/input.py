try:
    import anndata
except ImportError:
    anndata = None
from typing import List, Optional, Union

import dask.array
import numpy as np
import pandas as pd
import patsy
import scipy.sparse

from .external import InputDataBase
from .utils import parse_constraints, parse_design


class InputDataGLM(InputDataBase):
    """
    Input data for Generalized Linear Models (GLMs).
    Inherits from batchglm.models.base.input.InputDataBase.
    Contains additional information that is specific to GLM's like design matrices and constraints.

    Attributes
    ----------
    design_loc: Union[np.ndarray, pd.DataFrame, patsy.design_info.DesignMatrix]
        The location design model.
    design_scale:  Union[np.ndarray, pd.DataFrame, patsy.design_info.DesignMatrix]
        The scale design model.
    constraints_loc: np.ndarray
        Tensor that encodes how complete parameter set which includes dependent
        parameters arises from indepedent parameters: all = <constraints, indep>.
        This tensor describes this relation for the mean model.
        This form of constraints is used in vector generalized linear models (VGLMs).
    constraints_scale: np.ndarray
        Tensor that encodes how complete parameter set which includes dependent
        parameters arises from indepedent parameters: all = <constraints, indep>.
        This tensor describes this relation for the dispersion model.
        This form of constraints is used in vector generalized linear models (VGLMs).
    size_factors: np.ndarray
        Constant scale factors of the mean model in the linker space.
    """

    def __init__(
        self,
        data: Union[np.ndarray, anndata.AnnData, anndata.Raw, scipy.sparse.csr_matrix, dask.array.core.Array],
        design_loc: Optional[
            Union[np.ndarray, pd.DataFrame, patsy.design_info.DesignMatrix, dask.array.core.Array]
        ] = None,
        design_loc_names: Optional[Union[list, np.ndarray]] = None,
        design_scale: Optional[
            Union[np.ndarray, pd.DataFrame, patsy.design_info.DesignMatrix, dask.array.core.Array]
        ] = None,
        design_scale_names: Optional[Union[list, np.ndarray]] = None,
        constraints_loc: Optional[Union[np.ndarray, dask.array.core.Array]] = None,
        constraints_scale: Optional[Union[np.ndarray, dask.array.core.Array]] = None,
        size_factors=None,
        observation_names=None,
        feature_names=None,
        chunk_size_cells: int = 1000000,
        chunk_size_genes: int = 100,
        as_dask: bool = True,
        cast_dtype: str = "float64",
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
            chunk_size_cells=chunk_size_cells,
            chunk_size_genes=chunk_size_genes,
            cast_dtype=cast_dtype,
            as_dask=as_dask,
        )

        design_loc, design_loc_names = parse_design(design_matrix=design_loc, param_names=design_loc_names)
        design_scale, design_scale_names = parse_design(design_matrix=design_scale, param_names=design_scale_names)

        if as_dask:
            self.design_loc = dask.array.from_array(
                design_loc.astype(cast_dtype if cast_dtype is not None else self.x.dtype),
                chunks=(chunk_size_cells, 1000),
            )
            self.design_scale = dask.array.from_array(
                design_scale.astype(cast_dtype if cast_dtype is not None else self.x.dtype),
                chunks=(chunk_size_cells, 1000),
            )
        else:
            self.design_loc = design_loc.astype(cast_dtype if cast_dtype is not None else self.x.dtype)
            self.design_scale = design_scale.astype(cast_dtype if cast_dtype is not None else self.x.dtype)
        self._design_loc_names = design_loc_names
        self._design_scale_names = design_scale_names

        constraints_loc, loc_names = parse_constraints(
            dmat=design_loc, dmat_par_names=design_loc_names, constraints=constraints_loc, constraint_par_names=None
        )
        constraints_scale, scale_names = parse_constraints(
            dmat=design_scale,
            dmat_par_names=design_scale_names,
            constraints=constraints_scale,
            constraint_par_names=None,
        )
        if as_dask:
            self.constraints_loc = dask.array.from_array(
                constraints_loc.astype(cast_dtype if cast_dtype is not None else self.x.dtype),
                chunks=(1000, 1000),
            )
            self.constraints_scale = dask.array.from_array(
                constraints_scale.astype(cast_dtype if cast_dtype is not None else self.x.dtype),
                chunks=(1000, 1000),
            )
        else:
            self.constraints_loc = constraints_loc.astype(cast_dtype if cast_dtype is not None else self.x.dtype)
            self.constraints_scale = constraints_scale.astype(cast_dtype if cast_dtype is not None else self.x.dtype)
        self._loc_names = loc_names
        self._scale_names = scale_names

        if size_factors is not None:
            if len(size_factors.shape) == 1:
                size_factors = np.expand_dims(np.asarray(size_factors), axis=-1)
            elif len(size_factors.shape) == 2:
                pass
            else:
                raise ValueError("received size factors with dimension=%i" % len(size_factors.shape))
        if as_dask:
            self.size_factors = (
                dask.array.from_array(
                    size_factors.astype(cast_dtype if cast_dtype is not None else self.x.dtype),
                    chunks=(chunk_size_cells, 1),
                )
                if size_factors is not None
                else None
            )
        else:
            self.size_factors = (
                size_factors.astype(cast_dtype if cast_dtype is not None else self.x.dtype)
                if size_factors is not None
                else None
            )

    @property
    def design_loc_names(self):
        """Names of the location design matrix columns"""
        return self._design_loc_names

    @property
    def design_scale_names(self):
        """Names of the scale design matrix columns"""
        return self._design_scale_names

    @property
    def loc_names(self):
        """Names of the location design matrix columns subject to constraints"""
        return self._loc_names

    @property
    def scale_names(self):
        """Names of the scale design matrix columns subject to constraints"""
        return self._scale_names

    @property
    def num_design_loc_params(self):
        """Number of columns of the location design matrix"""
        return self.design_loc.shape[1]

    @property
    def num_design_scale_params(self):
        """Number of columns of the scale design matrix"""
        return self.design_scale.shape[1]

    @property
    def num_loc_params(self):
        """Number of columns of the location design matrix subject to constraints"""
        return self.constraints_loc.shape[1]

    @property
    def num_scale_params(self):
        """Number of columns of the scale design matrix subject to constraints"""
        return self.constraints_scale.shape[1]

    def fetch_design_loc(
        self, idx: Union[np.ndarray, List[bool]]
    ) -> Union[np.ndarray, pd.DataFrame, patsy.design_info.DesignMatrix]:
        """
        Obtain a selection of observations from the location design matrix.
        :param idx: A boolean mask to index a subset selection from the location design matrix
        :returns: Requested rows of the location design matrix
        """
        return self.design_loc[idx, :]

    def fetch_design_scale(self, idx):
        """
        Obtain a selection of observations from the scale design matrix.
        :param idx: A boolean mask to index a subset selection from the scale design matrix
        :returns: Requested rows of the scale design matrix
        """
        return self.design_scale[idx, :]

    def fetch_size_factors(self, idx):
        """
        Obtain a selection of size factors from the size factors matrix.
        :param idx: A boolean mask to index a subset selection from the size factors matrix
        :returns: Requested rows of the size factor matrix
        """
        return self.size_factors[idx, :]
