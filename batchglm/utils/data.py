import logging
from typing import List, Optional, Tuple, Union
from functools import singledispatch

import numpy as np
import pandas as pd
import patsy

try:
    import anndata

    try:
        from anndata.base import Raw
    except ImportError:
        from anndata import Raw
except ImportError:
    anndata = None
    Raw = None

logger = logging.getLogger("batchglm")


def design_matrix(
    sample_description: Optional[pd.DataFrame] = None,
    formula: Optional[str] = None,
    as_categorical: Union[bool, list] = True,
    dmat: Union[pd.DataFrame, None] = None,
    return_type: str = "patsy",
) -> Tuple[Union[patsy.design_info.DesignMatrix, pd.DataFrame], List[str]]:
    """
    Create a design matrix from some sample description.

    This function defaults to perform formatting if dmat is directly supplied as a pd.DataFrame.

    :param sample_description: pandas.DataFrame of length "num_observations" containing explanatory variables as
        columns. Ignored if dmat is provided.
    :param formula: model formula as string, describing the relations of the explanatory variables.

        E.g. '~ 1 + batch + confounder'
    :param as_categorical: boolean or list of booleans corresponding to the columns in 'sample_description'

        If True, all values in 'sample_description' will be treated as categorical values.

        If list of booleans, each column will be changed to categorical if the corresponding value in 'as_categorical'
        is True.

        Set to false, if columns should not be changed.
    :param dmat: a model design matrix as a pd.DataFrame
    :param return_type: type of the returned value.

        - "patsy": return plain patsy.design_info.DesignMatrix object
        - "dataframe": return pd.DataFrame with observations as rows and params as columns
    :return: a model design matrix
    """
    if dmat is None:
        if sample_description is None:
            raise ValueError("Provide a sample_description if dmat is None.")
        if isinstance(as_categorical, bool):
            as_categorical = [as_categorical] * sample_description.columns.size
        sample_description = sample_description.copy()
        columns = sample_description.columns[as_categorical]
        sample_description[columns] = sample_description[columns].apply(lambda col: col.astype("category"))

        dmat = patsy.dmatrix(formula, sample_description)
        coef_names = dmat.design_info.column_names

        if return_type == "dataframe":
            df = pd.DataFrame(dmat, columns=dmat.design_info.column_names)
            df = pd.concat([df, sample_description], axis=1)
            df.set_index(list(sample_description.columns), inplace=True)

            return df
        elif return_type == "patsy":
            return dmat, coef_names
        else:
            raise ValueError("return type %s not recognized" % return_type)
    else:
        if sample_description is not None:
            logger.warning("Both dmat and sample_description were given. Ignoring sample_description.")
        if return_type == "dataframe":
            return dmat, dmat.columns
        elif return_type == "patsy":
            raise ValueError("return type 'patsy' not supported for input (dmat is not None)")
        else:
            raise ValueError("return type %s not recognized" % return_type)


def view_coef_names(dmat: Union[patsy.design_info.DesignMatrix, pd.DataFrame]) -> np.ndarray:
    """
    Show names of coefficient in dmat.

    This wrapper provides quick access to this object attribute across all supported frameworks.

    :param dmat: Design matrix.
    :return: Array of coefficient names.
    """
    if isinstance(dmat, pd.DataFrame):
        return np.asarray(dmat.columns)
    elif isinstance(dmat, patsy.design_info.DesignMatrix):
        return np.asarray(dmat.design_info.column_names)
    else:
        raise ValueError("dmat type %s not recognized" % type(dmat))


def preview_coef_names(
    sample_description: pd.DataFrame, formula: str, as_categorical: Union[bool, list] = True
) -> List[str]:
    """
    Return coefficient names of model.

    Use this to preview what the model would look like.

    :param sample_description: pandas.DataFrame of length "num_observations" containing explanatory variables as columns
    :param formula: model formula as string, describing the relations of the explanatory variables.

        E.g. '~ 1 + batch + confounder'
    :param as_categorical: boolean or list of booleans corresponding to the columns in 'sample_description'

        If True, all values in 'sample_description' will be treated as categorical values.

        If list of booleans, each column will be changed to categorical if the corresponding value in 'as_categorical'
        is True.

        Set to false, if columns should not be changed.
    :return: A list of coefficient names.
    """
    _, coef_names = design_matrix(
        sample_description=sample_description, formula=formula, as_categorical=as_categorical, return_type="patsy"
    )
    return coef_names


@singledispatch
def constraint_system_from_star(
    constraints,
    dmat: Optional[Union[patsy.design_info.DesignMatrix, pd.DataFrame]] = None,
    sample_description: Optional[pd.DataFrame] = None,
    formula: Optional[str] = None,
    as_categorical: Union[bool, list] = True,
    return_type: str = "patsy",
) -> Tuple:
    """
    Wrap different constraint matrix building formats with building of design matrix.
    :param constraints: Constraints for model. Can be one of the following:

        - np.ndarray:
            Array with constraints in rows and model parameters in columns.
            Each constraint contains non-zero entries for the a of parameters that
            has to sum to zero. This constraint is enforced by binding one parameter
            to the negative sum of the other parameters, effectively representing that
            parameter as a function of the other parameters. This dependent
            parameter is indicated by a -1 in this array, the independent parameters
            of that constraint (which may be dependent at an earlier constraint)
            are indicated by a 1. You should only use this option
            together with prebuilt design matrix for the scale model, dmat_scale,
            for example via de.utils.setup_constrained().
        - dict:
            Every element of the dictionary corresponds to one set of equality constraints.
            Each set has to be be an entry of the form {..., x: y, ...}
            where x is the factor to be constrained and y is a factor by which levels of x are grouped
            and then constrained. Set y="1" to constrain all levels of x to sum to one,
            a single equality constraint.

                E.g.: {"batch": "condition"} Batch levels within each condition are constrained to sum to
                    zero. This is applicable if repeats of a an experiment within each condition
                    are independent so that the set-up ~1+condition+batch is perfectly confounded.

            Can only group by non-constrained effects right now, use constraint_matrix_from_string
            for other cases.
        - list of strings:
            String encoded equality constraints.

                E.g. ["batch1 + batch2 + batch3 = 0"]
        - None:
            No constraints are used, this is equivalent to using an identity matrix as a
            constraint matrix.
    :param dmat: Pre-built model design matrix.
    :param sample_description: pandas.DataFrame of length "num_observations" containing explanatory variables as columns
    :param formula: model formula as string, describing the relations of the explanatory variables.

        E.g. '~ 1 + batch + confounder'
        Only required if
    :param as_categorical: boolean or list of booleans corresponding to the columns in 'sample_description'

        If True, all values in 'sample_description' will be treated as categorical values.

        If list of booleans, each column will be changed to categorical if the corresponding value in 'as_categorical'
        is True.

        Set to false, if columns should not be changed.
    :param return_type: type of the returned value.

        - "patsy": return plain patsy.design_info.DesignMatrix object
        - "dataframe": return pd.DataFrame with observations as rows and params as columns

        This option is overridden if constraints are supplied as dict.
    :return: a model design matrix and a constraint matrix
    """
    if constraints is None:
        cmat = None
        term_names = None
        if isinstance(dmat, pd.DataFrame):
            coef_names = dmat.columns
            dmat = dmat.values
        else:
            coef_names = dmat.design_info.column_names
    else:
        raise ValueError("constraint format %s not recognized" % type(constraints))

    # Test full design matrix for being full rank before returning:
    if cmat is None:
        if np.linalg.matrix_rank(dmat) != dmat.shape[1]:
            raise ValueError(
                "constrained design matrix is not full rank: %i %i" % (np.linalg.matrix_rank(dmat), dmat.shape[1])
            )
    else:
        if np.linalg.matrix_rank(np.matmul(dmat, cmat)) != cmat.shape[1]:
            raise ValueError(
                "constrained design matrix is not full rank: %i %i"
                % (np.linalg.matrix_rank(np.matmul(dmat, cmat)), cmat.shape[1])
            )

    return dmat, coef_names, cmat, term_names

def _assert_design_mat_full_rank(cmat, dmat):
    # Test full design matrix for being full rank before returning:
    if cmat is None:
        assert np.linalg.matrix_rank(dmat) == dmat.shape[1], "constrained design matrix is not full rank: %i %i" % (np.linalg.matrix_rank(dmat), dmat.shape[1])
    else:
        assert np.linalg.matrix_rank(np.matmul(dmat, cmat)) == cmat.shape[1],  "constrained design matrix is not full rank: %i %i" % (np.linalg.matrix_rank(np.matmul(dmat, cmat)), cmat.shape[1])

@constraint_system_from_star.register
def _(
    constraints: dict,
    dmat: Optional[Union[patsy.design_info.DesignMatrix, pd.DataFrame]] = None,
    sample_description: Optional[pd.DataFrame] = None,
    formula: Optional[str] = None,
    as_categorical: Union[bool, list] = True,
    return_type: str = "patsy",
):
    if sample_description is None and dmat is None:
        raise ValueError("supply either sample_description or dmat")
    if dmat is not None and isinstance(constraints, dict):
        raise ValueError("dmat was supplied even though constraints were given as dict")
    if formula is None:
        raise ValueError("Provide a formula when providing constraints as dict.")
    dmat, coef_names, cmat, term_names = constraint_matrix_from_dict(
        sample_description=sample_description,
        formula=formula,
        as_categorical=as_categorical,
        constraints=constraints,
        return_type="patsy",
    )
    _assert_design_mat_full_rank(cmat, dmat)

    return dmat, coef_names, cmat, term_names

@constraint_system_from_star.register
def _(
    constraints: list, # something wrong with using tuples here
    dmat: Optional[Union[patsy.design_info.DesignMatrix, pd.DataFrame]] = None,
    sample_description: Optional[pd.DataFrame] = None,
    formula: Optional[str] = None,
    as_categorical: Union[bool, list] = True,
    return_type: str = "patsy",
):
    if sample_description is None and dmat is None:
        raise ValueError("supply either sample_description or dmat")
    if dmat is None:
        dmat, coef_names = design_matrix(
            sample_description=sample_description,
            formula=formula,
            as_categorical=as_categorical,
            dmat=None,
            return_type=return_type,
        )
    cmat = constraint_matrix_from_string(
        dmat=dmat, coef_names=dmat.design_info.column_names, constraints=constraints
    )
    term_names = None  # not supported yet.
    _assert_design_mat_full_rank(cmat, dmat)

    return dmat, coef_names, cmat, term_names

@constraint_system_from_star.register
def _(
    constraints: np.ndarray,
    dmat: Optional[Union[patsy.design_info.DesignMatrix, pd.DataFrame]] = None,
    sample_description: Optional[pd.DataFrame] = None,
    formula: Optional[str] = None,
    as_categorical: Union[bool, list] = True,
    return_type: str = "patsy",
):
    if sample_description is None and dmat is None:
        raise ValueError("supply either sample_description or dmat")
    cmat = constraints
    term_names = None
    if dmat is None:
        dmat, coef_names = design_matrix(
            sample_description=sample_description,
            formula=formula,
            as_categorical=as_categorical,
            dmat=None,
            return_type=return_type,
        )
    if isinstance(dmat, pd.DataFrame):
        coef_names = dmat.columns
        dmat = dmat.values
    else:
        coef_names = dmat.design_info.column_names
    _assert_design_mat_full_rank(cmat, dmat)

    return dmat, coef_names, cmat, term_names


def constraint_matrix_from_dict(
    sample_description: pd.DataFrame,
    formula: str,
    as_categorical: Union[bool, list] = True,
    constraints: Optional[dict] = None,
    return_type: str = "patsy",
) -> Tuple:
    """
    Create a design matrix from some sample description and a constraint matrix
    based on factor encoding of constrained parameter sets.

    Note that we build a dataframe instead of a pasty.DesignMatrix here if constraints are used.
    This is done because we were not able to build a patsy.DesignMatrix of the constrained form
    required in this context. In those cases in which the return type cannot be patsy, we encourage the
    use of the returned term_names to perform term-wise slicing which is not supported by other
    design matrix return types.

    :param sample_description: pandas.DataFrame of length "num_observations" containing explanatory variables as columns
    :param formula: model formula as string, describing the relations of the explanatory variables.

        E.g. '~ 1 + batch + confounder'
    :param as_categorical: boolean or list of booleans corresponding to the columns in 'sample_description'

        If True, all values in 'sample_description' will be treated as categorical values.

        If list of booleans, each column will be changed to categorical if the corresponding value in 'as_categorical'
        is True.

        Set to false, if columns should not be changed.
    :param constraints: Grouped factors to enfore equality constraints on. Every element of
        the dictionary corresponds to one set of equality constraints. Each set has to be
        be an entry of the form {..., x: y, ...} where x is the factor to be constrained and y is
        a factor by which levels of x are grouped and then constrained. Set y="1" to constrain
        all levels of x to sum to one, a single equality constraint.

            E.g.: {"batch": "condition"} Batch levels within each condition are constrained to sum to
                zero. This is applicable if repeats of a an experiment within each condition
                are independent so that the set-up ~1+condition+batch is perfectly confounded.

        Can only group by non-constrained effects right now, use constraint_matrix_from_string
        for other cases.
    :return:
        - model design matrix
        - term_names to allow slicing by factor if return type cannot be patsy.DesignMatrix
    """
    if constraints is None:
        constraints = {}
    assert len(constraints) > 0, "supply constraints"
    sample_description: pd.DataFrame = sample_description.copy()

    if type(as_categorical) is not bool or as_categorical:
        if type(as_categorical) is bool and as_categorical:
            as_categorical = np.repeat(True, sample_description.columns.size)

        for to_cat, col in zip(as_categorical, sample_description):
            if to_cat:
                sample_description[col] = sample_description[col].astype("category")

    # Build core design matrix on unconstrained factors. Then add design matrices without
    # absorption of the first level of each factor for each constrained factor onto the
    # core matrix.
    formula_unconstrained = formula.split("+")
    formula_unconstrained = [x for x in formula_unconstrained if x.strip(" ") not in constraints.keys()]
    formula_unconstrained = "+".join(formula_unconstrained)
    dmat = patsy.dmatrix(formula_unconstrained, sample_description)
    coef_names = dmat.design_info.column_names
    term_names = dmat.design_info.term_names

    constraints_ls = string_constraints_from_dict(sample_description=sample_description, constraints=constraints)
    for x in constraints.keys():
        assert isinstance(x, str), "constrained should contain strings"
        dmat_constrained_temp = patsy.highlevel.dmatrix("0+" + x, sample_description)
        dmat = np.hstack([dmat, dmat_constrained_temp])
        coef_names.extend(dmat_constrained_temp.design_info.column_names)
        term_names.extend(dmat_constrained_temp.design_info.term_names)

    # Build constraint matrix.
    constraints_ar = constraint_matrix_from_string(dmat=dmat, coef_names=coef_names, constraints=constraints_ls)

    # Format return type
    if return_type == "dataframe":
        dmat = pd.DataFrame(dmat, columns=coef_names)
    return dmat, coef_names, constraints_ar, term_names


def string_constraints_from_dict(sample_description: pd.DataFrame, constraints: Optional[dict] = None):
    r"""
    Create string-encoded constraints from dictionary encoded constraints and sample description.

    :param sample_description: pandas.DataFrame of length "num_observations" containing explanatory variables as columns
    :param constraints: Grouped factors to enfore equality constraints on. Every element of
        the dictionary corresponds to one set of equality constraints. Each set has to be
        be an entry of the form {..., x: y, ...} where x is the factor to be constrained and y is
        a factor by which levels of x are grouped and then constrained. Set y="1" to constrain
        all levels of x to sum to one, a single equality constraint.

            E.g.: {"batch": "condition"} Batch levels within each condition are constrained to sum to
                zero. This is applicable if repeats of a an experiment within each condition
                are independent so that the set-up ~1+condition+batch is perfectly confounded.

        Can only group by non-constrained effects right now, use constraint_matrix_from_string
        for other cases.
    :return: List of constraints as strings.

        E.g. ["batch1 + batch2 + batch3 = 0"]
    """
    if constraints is not None:
        constraints_ls = []
        for i, x in enumerate(constraints.keys()):
            assert isinstance(x, str), "constrained should contain strings"
            dmat_constrained_temp = patsy.highlevel.dmatrix("0+" + x, sample_description)

            dmat_grouping_temp = patsy.highlevel.dmatrix("0+" + list(constraints.values())[i], sample_description)
            for j in range(dmat_grouping_temp.shape[1]):
                grouping = dmat_grouping_temp[:, j]
                idx_constrained_group = np.where(np.sum(dmat_constrained_temp[grouping == 1, :], axis=0) > 0)[0]
                # Assert that required grouping is nested.
                assert np.all(
                    np.logical_xor(
                        np.sum(dmat_constrained_temp[grouping == 1, :], axis=0) > 0,
                        np.sum(dmat_constrained_temp[grouping == 0, :], axis=0) > 0,
                    )
                ), "proposed grouping of constraints is not nested, read docstrings"
                # Add new string-encoded equality constraint.
                constraints_ls.append(
                    "+".join(list(np.asarray(dmat_constrained_temp.design_info.column_names)[idx_constrained_group]))
                    + "=0"
                )

        logging.getLogger("batchglm").warning("Built constraints: " + ", ".join(constraints_ls))
    else:
        constraints_ls = None

    return constraints_ls


def constraint_matrix_from_string(dmat: np.ndarray, coef_names: list, constraints: Union[Tuple[str, str], List[str]]):
    r"""
    Create constraint matrix form string encoded equality constraints.

    :param dmat: Design matrix.
    :param constraints: List of constraints as strings.

        E.g. ["batch1 + batch2 + batch3 = 0"]
    :return: a constraint matrix
    """
    assert len(constraints) > 0, "supply constraints"

    n_par_all = dmat.shape[1]
    n_par_free = n_par_all - len(constraints)

    di = patsy.DesignInfo(coef_names)
    constraint_ls = [di.linear_constraint(x).coefs[0] for x in constraints]
    # Check that constraints are sensible:
    for constraint_i in constraint_ls:
        if np.sum(constraint_i != 0) == 1:
            raise ValueError("a zero-equality constraint only involved one parameter: remove this parameter")
    idx_constr = np.asarray([np.where(x == 1)[0][0] for x in constraint_ls])
    idx_depending = [np.where(x == 1)[0][1:] for x in constraint_ls]
    idx_unconstr = np.asarray(list(set(np.asarray(range(n_par_all))) - set(idx_constr)))

    constraint_mat = np.zeros([n_par_all, n_par_free])
    for i in range(n_par_all):
        if i in idx_constr:
            idx_dep_i = idx_depending[np.where(idx_constr == i)[0][0]]
            idx_dep_i = np.asarray([np.where(idx_unconstr == x)[0] for x in idx_dep_i])
            constraint_mat[i, :] = 0
            constraint_mat[i, idx_dep_i] = -1
        else:
            idx_unconstr_i = np.where(idx_unconstr == i)
            constraint_mat[i, :] = 0
            constraint_mat[i, idx_unconstr_i] = 1

    # Test unconstrained subset design matrix for being full rank before returning constraints:
    if np.linalg.matrix_rank(dmat[:, idx_unconstr]) != np.linalg.matrix_rank(dmat[:, idx_unconstr].T):
        raise ValueError(
            "unconstrained sub-design matrix is not full rank" % np.linalg.matrix_rank(dmat[:, idx_unconstr]),
            np.linalg.matrix_rank(dmat[:, idx_unconstr].T),
        )

    return constraint_mat


def bin_continuous_covariate(
    sample_description: pd.DataFrame, factor_to_bin: str, bins: Union[int, list, np.ndarray, Tuple]
):
    r"""
    Bin a continuous covariate.

    Adds the binned covariate to the table. Binning is performed on quantiles of the distribution.

    :param sample_description: Sample description table.
    :param factor_to_bin: Name of columns of factor to bin.
    :param bins: Number of bins or iteratable with bin borders. If given as integer, the bins are defined on the
        quantiles of the covariate, ie the bottom 20% of observations are in the first bin if bins==5.
    :return: Sample description table with binned covariate added.
    """
    if isinstance(bins, list) or isinstance(bins, np.ndarray) or isinstance(bins, Tuple):
        bins = np.asarray(bins)
    else:
        bins = np.arange(0, 1, 1 / bins)

    sample_description[factor_to_bin + "_binned"] = np.digitize(
        np.argsort(np.argsort(sample_description[factor_to_bin].values)) / sample_description.shape[0], bins
    )
    return sample_description
