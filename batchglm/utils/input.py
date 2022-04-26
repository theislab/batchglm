import logging
from typing import List, Optional, Union, Tuple

import anndata
import dask.array
import numpy as np
import pandas as pd
import patsy
import scipy.sparse
import sparse

logger = logging.getLogger("batchglm")


def parse_design(
    design_matrix: Union[pd.DataFrame, patsy.design_info.DesignMatrix, dask.array.core.Array, np.ndarray],
    param_names: List[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    r"""
    Parser for design matrices.

    :param design_matrix: Design matrix.
    :param param_names:
        Optional coefficient names for design_matrix.
        Ignored if design_matrix is pd.DataFrame or patsy.design_info.DesignMatrix.
    :return: Tuple[np.ndarray, List[str]] containing the design matrix and the parameter names.
    :raise AssertionError: if the type of design_matrix is not understood.
    :raise AssertionError: if length of provided param_names is not equal to number of coefficients in design_matrix.
    :raise ValueError: if param_names is None when type of design_matrix is numpy.ndarray or dask.array.core.Array.
    """
    if isinstance(design_matrix, (pd.DataFrame, patsy.design_info.DesignMatrix)) and param_names is not None:
        logger.warning(f"The provided param_names are ignored as the design matrix is of type {type(design_matrix)}.")

    if isinstance(design_matrix, patsy.design_info.DesignMatrix):
        dmat = np.asarray(design_matrix)
        params = design_matrix.design_info.column_names
    elif isinstance(design_matrix, pd.DataFrame):
        dmat = np.asarray(design_matrix)
        params = design_matrix.columns.tolist()
    elif isinstance(design_matrix, dask.array.core.Array):
        dmat = design_matrix.compute()
        params = param_names
    elif isinstance(design_matrix, np.ndarray):
        dmat = design_matrix
        params = param_names
    else:
        raise AssertionError(f"Datatype for design_matrix not understood: {type(design_matrix)}")
    if params is None:
        raise ValueError("Provide names when passing design_matrix as np.ndarray or dask.array.core.Array!")
    assert len(params) == dmat.shape[1], (
        "Length of provided param_names is not equal to " "number of coefficients in design_matrix."
    )
    return dmat, params


def parse_constraints(
    dmat: np.ndarray,
    dmat_par_names: List[str],
    constraints: Optional[Union[np.ndarray, dask.array.core.Array]] = None,
    constraint_par_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    r"""
    Parser for constraint matrices.

    :param dmat: Design matrix.
    :param constraints: Constraint matrix.
    :param constraint_par_names: Optional coefficient names for constraints.
    :return: Tuple[np.ndarray, List[str]] containing the constraint matrix and the parameter names.
    :raise AssertionError: if the type of given design / contraint matrix is not np.ndarray or dask.array.core.Array.
    """
    assert isinstance(dmat, np.ndarray), "dmat must be provided as np.ndarray."
    if constraints is None:
        constraints = np.identity(n=dmat.shape[1])
        constraint_params = dmat_par_names
    else:
        if isinstance(constraints, dask.array.core.Array):
            constraints = constraints.compute()
        assert isinstance(constraints, np.ndarray), "contraints must be np.ndarray or dask.array.core.Array."
        # Cannot use all parameter names if constraint matrix is not identity: Make up new ones.
        # Use variable names that can be mapped (unconstrained).
        if constraint_par_names is not None:
            assert len(constraint_params) == len(constraint_par_names)
            constraint_params = constraint_par_names
        else:
            constraint_params = [
                "var_" + str(i)
                if np.sum(constraints[:, i] != 0) > 1
                else dmat_par_names[np.where(constraints[:, i] != 0)[0][0]]
                for i in range(constraints.shape[1])
            ]
        assert constraints.shape[0] == dmat.shape[1], "constraint dimension mismatch"

    return constraints, constraint_params


class InputDataGLM:
    """
    Input data for Generalized Linear Models (GLMs).
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
    features : List[str]
        Names of the features' names
    observations : List[str]
        Names of the observations' names
    x : Union[np.ndarray, dask.array.core.Array, scipy.sparse.csr_matrix]
        An observations x features matrix-like object (see possible types).  Note that this can be dense or sparse.
    chunk_size_cells : int
        dask chunk size for cells
    chunk_size_genes : int
        dask chunk size for genes
    """

    features: List[str]
    observations: List[str]
    chunk_size_cells: int
    chunk_size_genes: int
    x: Union[dask.array.core.Array, scipy.sparse.spmatrix, np.ndarray]

    def __init__(
        self,
        data: Union[np.ndarray, anndata.AnnData, anndata.Raw, scipy.sparse.csr_matrix, dask.array.core.Array],
        design_loc: Optional[
            Union[np.ndarray, pd.DataFrame, patsy.design_info.DesignMatrix, dask.array.core.Array]
        ] = None,
        design_loc_names: Optional[List[str]] = None,
        design_scale: Optional[
            Union[np.ndarray, pd.DataFrame, patsy.design_info.DesignMatrix, dask.array.core.Array]
        ] = None,
        design_scale_names: Optional[List[str]] = None,
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
        self.observations = observation_names
        self.features = feature_names
        if (
            isinstance(data, np.ndarray)
            or isinstance(data, scipy.sparse.csr_matrix)
            or isinstance(data, dask.array.core.Array)
        ):
            self.x = data
        elif isinstance(data, anndata.AnnData) or isinstance(data, anndata.Raw):
            self.x = data.X
        else:
            raise ValueError("type of data %s not recognized" % type(data))

        if as_dask:
            if isinstance(self.x, dask.array.core.Array):
                self.x = self.x.compute()
            # Need to wrap dask around the COO matrix version of the sparse package if matrix is sparse.
            if isinstance(self.x, scipy.sparse.spmatrix):
                self.x = dask.array.from_array(
                    sparse.COO.from_scipy_sparse(self.x.astype(cast_dtype if cast_dtype is not None else self.x.dtype)),
                    chunks=(chunk_size_cells, chunk_size_genes),
                    asarray=False,
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

        self._cast_dtype = cast_dtype

    @property
    def cast_dtype(self):
        return self._cast_dtype

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

    @property
    def num_observations(self):
        """Number of observations derived from x."""
        return self.x.shape[0]

    @property
    def num_features(self):
        """Number of features derived from x."""
        return self.x.shape[1]

    @property
    def feature_isnonzero(self):
        """Boolean whether or not all features are zero"""
        return ~self._feature_allzero

    @property
    def feature_isallzero(self):
        """Boolean whether or not all features are zero"""
        return self._feature_allzero
