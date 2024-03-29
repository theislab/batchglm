import logging
from functools import singledispatch
from typing import Callable, Dict, List, Optional, Tuple, Union

import dask
import numpy as np
import pandas as pd
import patsy

logger = logging.getLogger("batchglm")


def dask_compute(func: Callable):
    def func_wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result.compute() if isinstance(result, dask.array.core.Array) else result

    return func_wrapper


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
            as_categorical = sample_description.columns
        sample_description = sample_description.copy()
        sample_description[as_categorical] = sample_description[as_categorical].apply(
            lambda col: col.astype("category")
        )

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


def _assert_design_mat_full_rank(cmat, dmat):
    # Test full design matrix for being full rank before returning:
    if cmat is None:
        assert np.linalg.matrix_rank(dmat) == dmat.shape[1], "constrained design matrix is not full rank: %i %i" % (
            np.linalg.matrix_rank(dmat),
            dmat.shape[1],
        )
    else:
        assert (
            np.linalg.matrix_rank(np.matmul(dmat, cmat)) == cmat.shape[1]
        ), "constrained design matrix is not full rank: %i %i" % (
            np.linalg.matrix_rank(np.matmul(dmat, cmat)),
            cmat.shape[1],
        )


@singledispatch
def constraint_system_from_star(
    constraints=None,
    dmat: Optional[Union[patsy.design_info.DesignMatrix, pd.DataFrame]] = None,
    term_names: Optional[List[str]] = None,
    **kwargs,
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
    elif isinstance(constraints, np.ndarray):
        cmat = constraints
    else:
        raise TypeError(f"Type {type(constraints)} not recognized for argument constraints.")
    if dmat is None:
        dmat, coef_names = design_matrix(**kwargs)
    if isinstance(dmat, pd.DataFrame):
        coef_names = dmat.columns.to_list()
        dmat = dmat.values
    elif isinstance(dmat, patsy.DesignMatrix):
        coef_names = dmat.design_info.column_names
    else:
        raise TypeError(f"Type {type(dmat)} not recognized for argument dmat.")
    if term_names is None:
        term_names = coef_names

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


@constraint_system_from_star.register
def _constraint_system_from_dict(
    constraints: dict,
    **kwargs,
) -> Tuple:
    if "return_type" in kwargs:
        return_type = kwargs.pop("return_type")
    else:
        return_type = "patsy"
    if "dmat" in kwargs:
        kwargs.pop("dmat")  # not sure why dmat was an argument here but some things expect it to be part of the API.
    cmat, dmat, term_names = constraint_system_from_dict(constraints, **kwargs)
    return constraint_system_from_star(cmat, dmat=dmat, return_type=return_type, term_names=term_names, **kwargs)


@constraint_system_from_star.register
def _constraint_system_from_list(
    constraints: list,
    dmat: Optional[Union[patsy.DesignMatrix, pd.DataFrame]] = None,
    return_type: str = "patsy",
    **kwargs,
) -> Tuple:
    if dmat is None:
        dmat, _ = design_matrix(**kwargs)
    if isinstance(dmat, pd.DataFrame):
        cmat = constraint_matrix_from_string(
            dmat=dmat.to_numpy(), coef_names=dmat.columns.to_list(), constraints=constraints
        )
    elif isinstance(dmat, patsy.DesignMatrix):
        cmat = constraint_matrix_from_string(
            dmat=dmat, coef_names=dmat.design_info.column_names, constraints=constraints
        )
    else:
        raise TypeError(f"Type {type(dmat)} not recognized for argument dmat.")

    return constraint_system_from_star(cmat, dmat=dmat, return_type=return_type, **kwargs)


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
    constraints_ls = []
    if constraints is not None:
        for key, value in constraints.items():
            assert isinstance(key, str), "constrained should contain strings"
            dmat_constrained_temp = patsy.highlevel.dmatrix("0+" + key, sample_description)

            dmat_grouping_temp = patsy.highlevel.dmatrix("0+" + value, sample_description)
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
    return constraints_ls


def constraint_system_from_dict(
    constraints: dict,
    sample_description: pd.DataFrame,
    formula: str,
    as_categorical: Union[bool, List[str], pd.Index, np.ndarray] = True,
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
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
    # input sanity checks
    assert len(constraints) > 0, "Constraints must not be empty."
    sample_description = sample_description.copy()
    if isinstance(as_categorical, bool) and as_categorical:
        as_categorical = sample_description.columns

    # make columns categorical
    sample_description[as_categorical] = sample_description[as_categorical].astype("category")

    # Build core design matrix on unconstrained factors. Then add design matrices without
    # absorption of the first level of each factor for each constrained factor onto the
    # core matrix.
    formula_unconstrained_list = formula.split("+")
    formula_unconstrained_list = [x for x in formula_unconstrained_list if x.strip(" ") not in constraints.keys()]
    formula_unconstrained = "+".join(formula_unconstrained_list)
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
    cmat = constraint_matrix_from_string(dmat=dmat, coef_names=coef_names, constraints=constraints_ls)
    dmat = pd.DataFrame(dmat, columns=coef_names)

    return cmat, dmat, term_names


def constraint_matrix_from_string(dmat: np.ndarray, coef_names: list, constraints: List[str]) -> np.ndarray:
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
    sample_description: pd.DataFrame, factor_to_bin: str, bins: Union[int, List[Union[int, float]], np.ndarray, Tuple]
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
    if isinstance(bins, int):
        bins = np.arange(0, 1, 1 / bins)
    if isinstance(bins, (list, np.ndarray, tuple)):
        bins = np.asarray(bins)
    else:
        raise TypeError(f"Type {type(bins)} recognized for argument bins.")
    sample_description[factor_to_bin + "_binned"] = np.digitize(
        np.argsort(np.argsort(sample_description[factor_to_bin].values)) / sample_description.shape[0], bins
    )
    return sample_description
