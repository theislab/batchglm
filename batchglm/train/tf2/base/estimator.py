from .external import pkg_constants, TrainingStrategies
from .model import ModelBase, LossBase

import numpy as np
import tensorflow as tf


class TFEstimator:
    model: ModelBase
    loss: LossBase

    def __init__(self, input_data, dtype):

        self._input_data = input_data
        self.dtype = dtype

    def _train(
            self,
            batched_model: bool,
            batch_size: int,
            optimizer_object: tf.keras.optimizers.Optimizer,
            optimizer_enum: TrainingStrategies,
            convergence_criteria: str,
            stopping_criteria: int,
            autograd: bool,
            featurewise: bool,
            benchmark: bool
    ):
        pass

    def fetch_fn(self, idx):
        pass
