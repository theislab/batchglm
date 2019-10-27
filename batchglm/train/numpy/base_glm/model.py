import abc
import numpy as np
import logging

logger = logging.getLogger("batchglm")


class ModelIwls:

    def __init__(
            self,
            model_vars
    ):
        self.model_vars = model_vars
        self.params = np.concatenate(
            [
                model_vars.init_a_clipped,
                model_vars.init_b_clipped,
            ],
            axis=0
        )

    @abc.abstractmethod
    def fim_weight(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def ll(self) -> np.ndarray:
        pass

    @property
    def fim(self):
        w = np.diag(self.fim_weight)
        return np.matmul(
            np.matmul(
                np.matmul(self.design_loc.T, self.contraints_loc),
                w
            ),
            np.matmul(self.design_loc, self.constraints_loc)
        )

    def jac(self):
        w = np.diag(self.fim_weight)
        y_bar = (self.x - self.mu) / self.mu
        return np.matmul(
                np.matmul(
                    np.matmul(self.design_loc.T, self. constraints_loc.T),
                    w
                ),
            y_bar
        )
