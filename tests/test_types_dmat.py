import unittest
from typing import List, Union

import dask
import numpy as np
import pandas as pd
import patsy

from batchglm.models.base_glm.utils import parse_design


def check_np_dask(dmat: Union[np.ndarray, dask.array.core.Array], params: List[str]) -> bool:
    parse_design(design_matrix=dmat, param_names=params)
    try:  # must produce ValueError
        parse_design(design_matrix=dmat, param_names=None)
        return False
    except ValueError as ve:
        if not str(ve) == "Provide names when passing design_matrix as np.ndarray or dask.array.core.Array!":
            raise
    try:  # must result in AssertionError
        parse_design(design_matrix=dmat, param_names=params[:-1])
        return False
    except AssertionError as ae:
        if not (
            str(ae) == "Length of provided param_names is not equal to number of coefficients in design_matrix."
            or str(ae).startswith("Datatype for design_matrix not understood")
        ):
            raise
    return True


def check_pd_patsy(dmat: Union[pd.DataFrame, patsy.design_info.DesignMatrix], params: List[str]) -> bool:
    _, ret_params = parse_design(design_matrix=dmat, param_names=None)
    if ret_params != params:
        return False

    # generate new coefs to test ignoring passed params
    new_coef_list = ["a", "b", "c"]

    # param_names should be ignored
    _, ret_params = parse_design(design_matrix=dmat, param_names=new_coef_list)
    if params != ret_params:
        return False
    # param_names should be ignored
    _, ret_params = parse_design(design_matrix=dmat, param_names=new_coef_list[:-1])
    if params != ret_params:
        return False
    return True


class TestParseDesign(unittest.TestCase):
    """
    Test various input data types for parsing of design and constraint matrices.
    The method "parse_design" in batchglm.models.base_glm.utils must return Tuple[np.ndarray, List[str]].
    It must fail if no param_names are passed or the length of param_names is not equal to the length of params.
    """

    def test_parse_design(self) -> bool:
        # create artificial data
        obs, coef = (500, 3)
        dmat = np.zeros(shape=(obs, coef))
        coef_list = ["Intercept", "coef_0", "coef_1"]

        # check np
        if not (check_np_dask(dmat=dmat, params=coef_list)):
            return False
        # check dask
        if not (check_np_dask(dmat=dask.array.from_array(dmat, chunks=(1000, 1000)), params=coef_list)):
            return False
        # check pd
        pd_coef = pd.DataFrame(dmat, columns=coef_list)
        if not (check_pd_patsy(dmat=pd_coef, params=coef_list)):
            return False
        # check patsy
        if not (check_pd_patsy(dmat=patsy.dmatrix("~1 + coef_0 + coef_1", pd_coef), params=coef_list)):
            return False
        return True


if __name__ == "__main__":
    unittest.main()
