from typing import Tuple, Union
import numpy as np
import sys

from .external import InputDataGLM, Model, EstimatorGlm
from .external import init_par

from .vars import ModelVars
from .model import ModelIwlsNb


class Estimator(EstimatorGlm):
    """
    Estimator for Generalized Linear Models (GLMs) with negative binomial noise.
    Uses the natural logarithm as linker function.
    """
    model: ModelIwlsNb

    def __init__(
            self,
            input_data: InputDataGLM,
            init_a: Union[np.ndarray, str] = "AUTO",
            init_b: Union[np.ndarray, str] = "AUTO",
            batch_size: Union[None, Tuple[int, int]] = None,
            quick_scale: bool = False,
            dtype="float64",
            **kwargs
    ):
        """
        Performs initialisation and creates a new estimator.

        :param input_data: InputDataGLM
            The input data
        :param init_a: (Optional)
            Low-level initial values for a. Can be:

            - str:
                * "auto": automatically choose best initialization
                * "random": initialize with random values
                * "standard": initialize intercept with observed mean
                * "init_model": initialize with another model (see `ìnit_model` parameter)
                * "closed_form": try to initialize with closed form
            - np.ndarray: direct initialization of 'a'
        :param init_b: (Optional)
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
        init_a, init_b, train_loc, train_scale = init_par(
            input_data=input_data,
            init_a=init_a,
            init_b=init_b,
            init_model=None
        )
        self._train_loc = train_loc
        self._train_scale = train_scale
        if quick_scale:
            self._train_scale = False
        sys.stdout.write("training location model: %s\n" % str(self._train_loc))
        sys.stdout.write("training scale model: %s\n" % str(self._train_scale))
        init_a = init_a.astype(dtype)
        init_b = init_b.astype(dtype)

        self.model_vars = ModelVars(
            init_a=init_a,
            init_b=init_b,
            constraints_loc=input_data.constraints_loc,
            constraints_scale=input_data.constraints_scale,
            chunk_size_genes=input_data.chunk_size_genes,
            dtype=dtype
        )
        model = ModelIwlsNb(
            input_data=input_data,
            model_vars=self.model_vars,
            compute_mu=self._train_loc,
            compute_r=not self._train_scale,
            dtype=dtype
        )
        super(Estimator, self).__init__(
            input_data=input_data,
            model=model,
            dtype=dtype
        )

    def get_model_container(
            self,
            input_data
    ):
        return Model(input_data=input_data)
