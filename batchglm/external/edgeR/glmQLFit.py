from typing import List, Optional, Tuple, Union

import numpy as np

from .aveLogCPM import calculate_avg_log_cpm
from .estimator import NBEstimator
from .external import InputDataGLM, NBModel
from .prior_df import calculate_prior_df


def glm_ql_fit(
    x: Union[NBModel, np.ndarray],
    dispersion: Union[np.ndarray, float],
    design: Optional[np.ndarray] = None,
    design_loc_names: Optional[List[str]] = None,
    offset: Optional[np.ndarray] = None,
    lib_size: Optional[np.ndarray] = None,
    size_factors: Optional[np.ndarray] = None,
    tol: float = 1e-6,  # TODO
    weights: Optional[np.ndarray] = None,
    abundance_trend: bool = True,
    ave_log_cpm: Optional[np.ndarray] = None,
    robust: bool = False,
    winsor_tail_p: Tuple[float, float] = (0.05, 0.1),
    **input_data_kwargs,
):
    """
    Fit a GLM and compute quasi-likelihood dispersions for each gene.
    """
    #       Original method docstring:
    #       Fits a GLM and computes quasi-likelihood dispersions for each gene.
    #       Davis McCarthy, Gordon Smyth, Yunshun Chen, Aaron Lun.
    #       Originally part of glmQLFTest, as separate function 15 September 2014. Last modified 4 April 2020.

    if isinstance(x, np.ndarray):
        if design is None:
            raise AssertionError("Provide design when x is not a model already.")
        if size_factors is None:
            size_factors = np.log(x.sum(axis=1))
        input_data = InputDataGLM(
            data=x,
            design_loc=design,
            design_loc_names=design_loc_names,
            size_factors=size_factors,
            design_scale=np.ones((x.shape[0], 1)),
            design_scale_names=["Intercept"],
            **input_data_kwargs,
        )
        model = NBModel(input_data)
    elif isinstance(x, NBModel):
        model = x
    else:
        raise TypeError(f"Type for argument x not understood: {type(x)}. Valid types are NBModel, np.ndarray")

    # estimator = NBEstimator(model, dispersion=dispersion)
    # estimator.train(maxit=250, tolerance=tol)
    # glmfit = glmFit(y, design=design, dispersion=dispersion, offset=offset, lib.size=lib.size, weights=weights,...)

    # Setting up the abundances.
    if abundance_trend:
        if ave_log_cpm is None:
            pass
            # big TODO
            # ave_log_cpm = calculate_avg_log_cpm(x=model.x, size_factors=TODO, dispersion=dispersion, weights=weights)
            # ave_log_cpm = aveLogCPM(y, lib.size=lib.size, weights=weights, dispersion=dispersion)
            # glmfit$AveLogCPM <- AveLogCPM
    else:
        ave_log_cpm = None

    return calculate_prior_df(
        model=model,
        avg_log_cpm=ave_log_cpm,
        robust=robust,
        winsor_tail_p=winsor_tail_p,
        dispersion=dispersion,
        tolerance=tol,
    )
