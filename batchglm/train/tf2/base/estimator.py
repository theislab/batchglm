from .model import ModelBase
import tensorflow as tf


class TFEstimator:
    model: ModelBase

    def __init__(self, input_data, dtype):

        self._input_data = input_data
        self.dtype = dtype

    def _train(
            self,
            is_batched: bool,
            batch_size: int,
            optimizer_object: tf.keras.optimizers.Optimizer,
            convergence_criteria: str,
            stopping_criteria: int,
            autograd: bool,
            featurewise: bool,
            benchmark: bool,
            optimizer: str
    ):
        pass

    def fetch_fn(self, idx):
        pass
