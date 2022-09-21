import sys
from typing import Optional, Tuple, Union

import numpy as np

from .external import EstimatorGlm, Model, init_par
from .model_container import ModelContainer


class Estimator(EstimatorGlm):
    """
    Estimator for Generalized Linear Models (GLMs) with negative binomial noise.
    Uses the natural logarithm as linker function.

    Attributes
    ----------
    model_vars : ModelVars
        model variables
    """

    def __init__(
        self,
        model: Model,
        init_location: str = "AUTO",
        init_scale: str = "AUTO",
        # batch_size: Optional[Union[Tuple[int, int], int]] = None,
        quick_scale: bool = False,
        dtype: str = "float64",
    ):
        """
        Performs initialisation and creates a new estimator.

        :param init_location: (Optional)
            Low-level initial values for a. Can be:

            - str:
                * "auto": automatically choose best initialization
                * "standard": initialize intercept with observed mean
                * "init_model": initialize with another model (see `ìnit_model` parameter)
                * "closed_form": try to initialize with closed form
            - np.ndarray: direct initialization of 'a'
        :param dtype: Numerical precision.
        """
        init_theta_location, _, train_loc, _ = init_par(model=model, init_location=init_location)
        self._train_loc = train_loc
        # no need to train the scale parameter for the poisson model since it only has one parameter
        self._train_scale = False
        sys.stdout.write("training location model: %s\n" % str(self._train_loc))
        init_theta_location = init_theta_location.astype(dtype)

        _model_container = ModelContainer(
            model=model,
            init_theta_location=init_theta_location,
            init_theta_scale=init_theta_location,  # Not used.
            chunk_size_genes=model.chunk_size_genes,
            dtype=dtype,
        )
        super(Estimator, self).__init__(model_container=_model_container, dtype=dtype)
