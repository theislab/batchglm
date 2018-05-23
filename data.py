from typing import Union

import patsy
import pandas as pd
import numpy as np
import xarray as xr


def design_matrix(sample_description: pd.DataFrame, formula: str,
                  as_categorical: Union[bool, list] = True) -> patsy.design_info.DesignMatrix:
    """
    Create a design matrix from some sample description
    
    :param sample_description: pandas.DataFrame containing explanatory variables as columns.
    :param formula: model formula as string, describing the relations of the explanatory variables.
    
        E.g. '~ 1 + batch + confounder'
    :param as_categorical: boolean or list of booleans corresponding to the columns in 'sample_description'
        
        If True, all values in 'sample_description' will be treated as categorical values.
        
        If list of booleans, each column will be changed to categorical if the corresponding value in 'as_categorical'
        is True.
        
        Set to false, if columns should not be changed.
        
    :return: a model design matrix
    """
    sample_description: pd.DataFrame = sample_description.copy()
    
    if type(as_categorical) is not bool or as_categorical:
        if type(as_categorical) is bool and as_categorical:
            as_categorical = np.repeat(True, sample_description.columns.size)
        
        for to_cat, col in zip(as_categorical, sample_description):
            if to_cat:
                sample_description[col] = sample_description[col].astype("category")
    
    dmatrix = patsy.dmatrix(formula, sample_description)
    return dmatrix


def design_matrix_from_dataset(dataset: xr.Dataset,
                               formula=None,
                               formula_key="formula",
                               explanatory_vars=None,
                               explanatory_vars_key="explanatory_vars",
                               design_key="design",
                               as_categorical=True,
                               inplace=False):
    """
    Create a design matrix from a given xarray.Dataset and model formula.
    
    The formula will be chosen by the following order:
        1) from the parameter 'formula'
        2) from dataset[formula_key]
        3) arbitrarily as '~ 1 {+ explanatory_vars[i]}' for all elements in the explanatory variables
    
    The explanatory variables will be chosen by the following order:
        1) from the parameter 'explanatory_vars'
        2) from dataset[explanatory_vars_key]
        3) all variables inside 'dataset'
    
    The resulting design matrix as well as the formula and explanatory variables will be stored at the corresponding
    '*_key' keys in the returned dataset.
    
    :param dataset: xarray.Dataset containing explanatory variables.
    :param formula: model formula as string, describing the relations of the explanatory variables.
        If None, the formula is assumed to be stored inside 'dataset' as attribute
    
        E.g. '~ 1 + batch + confounder'
    :param formula_key: index of the formula attribute inside 'dataset'.
        Only used, if 'formula' is None.
    :param explanatory_vars: list of variable keys to use as explanatory variables.
    :param explanatory_vars_key: index of the coordinate containing the explanatory variables.
        Only used, if 'explanatory_vars' is None.
    :param design_key: Under which key should the design matrix be stored?
    :param as_categorical: boolean or list of booleans corresponding to the columns in 'sample_description'
        
        If True, all values in 'sample_description' will be treated as categorical values.
        
        If list of booleans, each column will be changed to categorical if the corresponding value in 'as_categorical'
        is True.
        
        Set to false, if columns should not be changed.
    :param inplace: should 'dataset' be changed inplace?
    :return: xarray.Dataset containing the created design matrix as variable named 'design_key'
    """
    if not inplace:
        dataset = dataset.copy()
    
    if explanatory_vars is None:
        explanatory_vars = dataset.get(explanatory_vars_key)
    if explanatory_vars is None:
        explanatory_vars = list(dataset.variables.keys())
    
    if formula is None:
        formula = dataset.attrs.get(formula_key)
    if formula is None:
        formula = ["~ 1", ]
        for i in explanatory_vars:
            formula.append(" + " + i)
        formula = "".join(formula)
    
    dimensions = list(dataset[explanatory_vars].dims.keys())
    dimensions.append("design_params")
    
    sample_description = dataset[explanatory_vars].to_dataframe()
    dataset[design_key] = (
        dimensions,
        design_matrix(sample_description=sample_description, formula=formula, as_categorical=as_categorical)
    )
    
    dataset.attrs[formula_key] = formula
    dataset.coords[explanatory_vars_key] = explanatory_vars
    
    return dataset
