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
        **kwargs
    ):
        """
        Performs initialisation and creates a new estimator.

        :param init_location: (Optional)
            Low-level initial values for a. Can be:

            - str:
                * "auto": automatically choose best initialization
                * "random": initialize with random values
                * "standard": initialize intercept with observed mean
                * "init_model": initialize with another model (see `ìnit_model` parameter)
                * "closed_form": try to initialize with closed form
            - np.ndarray: direct initialization of 'a'
        :param init_scale: (Optional)
            Low-level initial values for b. Can be:

            - str:
                * "auto": automatically choose best initialization
                * "random": initialize with random values
                * "standard": initialize with zeros
                * "init_model": initialize with another model (see `ìnit_model` parameter)
                * "closed_form": try to initialize with closed form
            - np.ndarray: direct initialization of 'b'
        :param quick_scale: bool
            Whether `scale` will be fitted faster and maybe less accurate.
            Useful in scenarios where fitting the exact `scale` is not absolutely necessary.
        :param dtype: Numerical precision.
        """
        init_theta_location, init_theta_scale, train_loc, train_scale = init_par(
            model=model, init_location=init_location, init_scale=init_scale
        )
        self._train_loc = train_loc
        self._train_scale = train_scale
        if quick_scale:
            self._train_scale = False
        sys.stdout.write("training location model: %s\n" % str(self._train_loc))
        sys.stdout.write("training scale model: %s\n" % str(self._train_scale))
        init_theta_location = init_theta_location.astype(dtype)
        init_theta_scale = init_theta_scale.astype(dtype)

        _model_container = ModelContainer(
            model=model,
            init_theta_location=init_theta_location,
            init_theta_scale=init_theta_scale,
            chunk_size_genes=model.chunk_size_genes,
            dtype=dtype,
        )
        super(Estimator, self).__init__(model_container=_model_container, dtype=dtype)
