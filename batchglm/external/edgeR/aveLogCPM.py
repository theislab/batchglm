from typing import Optional, Union

import dask.array
import numpy as np

from .external import InputDataGLM, ModelContainer
from .glm_one_group import fit_single_group, get_single_group_start


def calculate_avg_log_cpm(
    x: np.ndarray,
    model_class,
    size_factors: Optional[np.ndarray] = None,
    dispersion: Union[np.ndarray, float] = 0.05,
    prior_count: int = 2,
    weights: Optional[np.ndarray] = None,
    maxit: int = 50,
    tolerance: float = 1e-10,
    chunk_size_cells=1e6,
    chunk_size_genes=1e6,
):
    """
    Computes average log2 counts per million per feature over all observations.
    The method is a python derivative of edgeR's aveLogCPM method.

    :param x: the counts data.
    :param model_class: the class object to use for creation of a model during the calculation
    :param size_factors: Optional size_factors. This is equivalent to edgeR's offsets.
    :param dispersion: Optional fixed dispersion parameter used during the calculation.
    :param prior_count: The count to be added to x prior to calculation.
    :param weights: Optional weights per feature (currently unsupported and ignored)
    :param: maxit: The max number of iterations during newton-raphson approximation.
    :param: tolerance: The minimal difference in change used as a stopping criteria during NR approximation.
    :param: chunk_size_cells: chunks used over the feature axis when using dask
    :param: chunk_size_genes: chunks used over the observation axis when using dask
    """

    if weights is None:
        weights = 1.0
    if isinstance(dispersion, float):
        dispersion = np.full((1, x.shape[1]), dispersion, dtype=float)
    if size_factors is None:
        size_factors = np.full((x.shape[0], 1), np.log(1.0))

    adjusted_prior, adjusted_size_factors = add_priors(prior_count, size_factors)
    x += adjusted_prior
    avg_cpm_model = model_class(
        InputDataGLM(
            data=x,
            design_loc=np.ones((x.shape[0], 1)),
            design_loc_names=np.array(["Intercept"]),
            size_factors=adjusted_size_factors,
            design_scale=np.ones((x.shape[0], 1)),
            design_scale_names=np.array(["Intercept"]),
            as_dask=isinstance(x, dask.array.core.Array),
            chunk_size_cells=chunk_size_cells,
            chunk_size_genes=chunk_size_genes,
        )
    )
    avg_cpm_model = ModelContainer(
        model=avg_cpm_model,
        init_theta_location=get_single_group_start(avg_cpm_model.x, avg_cpm_model.size_factors),
        init_theta_scale=np.log(1 / dispersion),
        chunk_size_genes=chunk_size_genes,
        dtype=x.dtype,
    )

    fit_single_group(avg_cpm_model, maxit=maxit, tolerance=tolerance)
    output = (avg_cpm_model.theta_location + np.log(1e6)) / np.log(2)

    return output


def add_priors(prior_count: int, size_factors: np.ndarray):

    factors = np.exp(size_factors)
    avg_factors = np.mean(factors)
    adjusted_priors = prior_count * factors / avg_factors
    adjusted_size_factors = np.log(factors + 2 * adjusted_priors)

    return adjusted_priors, adjusted_size_factors
