from .external import GLM
from .processModel import ProcessModel


class BetaGLM(GLM, ProcessModel):

    def __init__(
            self,
            model_vars,
            optimizer: str,
            compute_a: bool,
            compute_b: bool,
            use_gradient_tape: bool,
            dtype: str,
    ):
        super(BetaGLM, self).__init__(
            model_vars=model_vars,
            noise_module='glm_beta',
            optimizer=optimizer,
            compute_a=compute_a,
            compute_b=compute_b,
            use_gradient_tape=use_gradient_tape,
            dtype=dtype
        )
