import logging

import numpy as np

from .external import EstimatorGlm, Model
from .model_container import ModelContainer
from ....models.glm_norm.utils import init_par
logger = logging.getLogger("batchglm")


class Estimator(EstimatorGlm):


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
        :param model:
            The GLM model to be fit
        :param init_location: (Optional)
            Low-level initial values for a. Can be:

            - str:
                * "auto": automatically choose best initialization
                * "standard": initialize intercept with observed mean
                * "closed_form": try to initialize with closed form
            - np.ndarray: direct initialization of 'a'
        :param init_scale: (Optional)
            Low-level initial values for b. Can be:

            - str:
                * "auto": automatically choose best initialization
                * "random": initialize with random values
                * "standard": initialize with zeros
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
        init_theta_location = init_theta_location.astype(dtype)
        init_theta_scale = init_theta_scale.astype(dtype)
        self._train_scale = train_scale
        self._train_loc = train_loc
        if quick_scale:
            self._train_scale = False
        _model_container = ModelContainer(
            model=model,
            init_theta_location=init_theta_location,
            init_theta_scale=init_theta_scale,
            chunk_size_genes=model.chunk_size_genes,
            dtype=dtype,
        )
        super(Estimator, self).__init__(model_container=_model_container, dtype=dtype)

    def train(
        self,
        **kwargs,
    ):
        model = self._model_container.model
        if self._train_loc:
            theta_location, _, _, _ = np.linalg.lstsq(model.xh_loc, model.x)
            self._model_container.theta_location = theta_location
        self._train_loc = False
        super().train(**kwargs)
        self._train_loc = True
