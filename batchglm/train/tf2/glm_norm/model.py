import logging

from .layers import UnpackParams, LinearLoc, LinearScale, LinkerLoc, LinkerScale, Likelihood
from .layers_gradients import Jacobian, Hessian, FIM
from .external import GLM, LossGLM
from .processModel import ProcessModel

logger = logging.getLogger(__name__)


class NormGLM(GLM, ProcessModel):

    def __init__(
            self,
            model_vars,
            dtype,
            compute_a,
            compute_b,
            use_gradient_tape,
            optimizer
    ):
        self.compute_a = compute_a
        self.compute_b = compute_b

        super(NormGLM, self).__init__(
            model_vars=model_vars,
            unpack_params=UnpackParams(),
            linear_loc=LinearLoc(),
            linear_scale=LinearScale(),
            linker_loc=LinkerLoc(),
            linker_scale=LinkerScale(),
            likelihood=Likelihood(dtype),
            jacobian=Jacobian(
                model_vars=model_vars,
                compute_a=self.compute_a,
                compute_b=self.compute_b,
                dtype=dtype),
            hessian=Hessian(
                model_vars=model_vars,
                compute_a=self.compute_a,
                compute_b=self.compute_b,
                dtype=dtype),
            fim=FIM(
                model_vars=model_vars,
                compute_a=self.compute_a,
                compute_b=self.compute_b,
                dtype=dtype),
            use_gradient_tape=use_gradient_tape,
            optimizer=optimizer
        )


class LossGLMNorm(LossGLM):

    """
    Full class
    """
