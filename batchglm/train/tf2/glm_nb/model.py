import logging

from .external import LossGLM, GLM
from .layers import UnpackParams, LinearLoc, LinearScale, LinkerLoc, LinkerScale, Likelihood
from .layers_gradients import Jacobian, Hessian, FIM

from .processModel import ProcessModel

logger = logging.getLogger(__name__)


class NBGLM(GLM, ProcessModel):

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

        super(NBGLM, self).__init__(
            model_vars=model_vars,
            unpack_params=UnpackParams(),
            linear_loc=LinearLoc(),
            linear_scale=LinearScale(),
            linker_loc=LinkerLoc(),
            linker_scale=LinkerScale(),
            likelihood=Likelihood(dtype),
            jacobian=Jacobian(model_vars=model_vars, dtype=dtype),
            hessian=Hessian(model_vars=model_vars, dtype=dtype),
            fim=FIM(model_vars=model_vars, dtype=dtype),
            use_gradient_tape=use_gradient_tape,
            optimizer=optimizer
        )


class LossGLMNB(LossGLM):
    """
    Full class
    """
