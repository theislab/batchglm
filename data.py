from typing import Union, Dict

import os
import tempfile
import zipfile as zf
import scanpy.api as sc

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


def _factors_from_formula(formula):
    desc = patsy.ModelDesc.from_formula(formula)
    factors = set()
    for l in [list(t.factors) for t in desc.rhs_termlist]:
        for i in l:
            factors.add(i.name())
    
    return factors


def design_matrix_from_dataset(dataset: xr.Dataset,
                               formula=None,
                               formula_key="formula",
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
    
    if formula is None:
        formula = dataset.attrs.get(formula_key)
    if formula is None:
        # could not find formula; try to construct it from explanatory variables
        raise ValueError("formula could not be found")
        
    factors = _factors_from_formula(formula)
    explanatory_vars = set(dataset.variables.keys())
    explanatory_vars = list(explanatory_vars.intersection(factors))
    
    dimensions = list(dataset[explanatory_vars].dims.keys())
    dimensions.append("design_params")
    
    sample_description = dataset[explanatory_vars].to_dataframe()
    dataset[design_key] = (
        dimensions,
        design_matrix(sample_description=sample_description, formula=formula, as_categorical=as_categorical)
    )
    
    dataset.attrs[formula_key] = formula
    
    return dataset


def load_mtx_to_adata(path, cache=True):
    ad = sc.read(os.path.join(path, "matrix.mtx"), cache=cache).T
    
    files = os.listdir(os.path.join(path))
    for file in files:
        if file.startswith("genes"):
            delim = ","
            if file.endswith("tsv"):
                delim = "\t"
            
            tbl = pd.read_csv(os.path.join(path, file), header=None, sep=delim)
            ad.var = tbl
            ad.var_names = tbl[1]
        elif file.startswith("barcodes"):
            delim = ","
            if file.endswith("tsv"):
                delim = "\t"
            
            tbl = pd.read_csv(os.path.join(path, file), header=None, sep=delim)
            ad.obs = tbl
            ad.obs_names = tbl[0]
    ad.var_names_make_unique()
    return ad


def load_mtx_to_xarray(path):
    matrix = sc.read(os.path.join(path, "matrix.mtx"), cache=False).X.toarray()
    
    # retval = xr.Dataset({
    #     "sample_data": (["samples", "genes"], np.transpose(matrix)),
    # })
    
    retval = xr.DataArray(np.transpose(matrix), dims=("samples", "genes"))
    
    files = os.listdir(os.path.join(path))
    for file in files:
        if file.startswith("genes"):
            delim = ","
            if file.endswith("tsv"):
                delim = "\t"
            
            tbl = pd.read_csv(os.path.join(path, file), header=None, sep=delim)
            # retval["var"] = (["var_annotations", "genes"], np.transpose(tbl))
            for col_id in tbl:
                retval.coords["gene_annot%d" % col_id] = ("genes", tbl[col_id])
        elif file.startswith("barcodes"):
            delim = ","
            if file.endswith("tsv"):
                delim = "\t"
            
            tbl = pd.read_csv(os.path.join(path, file), header=None, sep=delim)
            # retval["obs"] = (["obs_annotations", "samples"], np.transpose(tbl))
            for col_id in tbl:
                retval.coords["sample_annot%d" % col_id] = ("samples", tbl[col_id])
    return retval


def load_recursive_mtx(dir_or_zipfile, target_format="xarray", cache=True) -> Dict[str, xr.DataArray]:
    dir_or_zipfile = os.path.expanduser(dir_or_zipfile)
    if dir_or_zipfile.endswith(".zip"):
        path = tempfile.mkdtemp()
        zip_ref = zf.ZipFile(dir_or_zipfile)
        zip_ref.extractall(path)
        zip_ref.close()
    else:
        path = dir_or_zipfile
    
    adatas = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if file == "matrix.mtx":
                if target_format.lower() == "xarray":
                    ad = load_mtx_to_xarray(root)
                elif target_format.lower() == "adata":
                    ad = load_mtx_to_adata(root, cache=cache)
                else:
                    raise RuntimeError("Unknown target format %s" % target_format)
                
                adatas[root[len(path) + 1:]] = ad
    
    return adatas
