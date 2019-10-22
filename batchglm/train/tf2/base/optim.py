import abc
import logging
import tensorflow as tf

logger = logging.getLogger("batchglm")


class OptimizerBase(tf.keras.optimizers.Optimizer, metaclass=abc.ABCMeta):

    def __init__(self, name):
        super(OptimizerBase, self).__init__(name=name)

    @abc.abstractmethod
    def _resource_apply_dense(self, grad, handle):
        pass

    @abc.abstractmethod
    def _resource_apply_sparse(self, grad, handle, apply_state):
        pass

    @abc.abstractmethod
    def _create_slots(self):
        pass

    """
    @property
    @abc.abstractmethod
    def vars(self):
        pass

    @property
    @abc.abstractmethod
    def gradients(self):
        return None

    @property
    @abc.abstractmethod
    def hessians(self):
        pass

    @property
    @abc.abstractmethod
    def fims(self):
        pass

    @abc.abstractmethod
    def step(self, learning_rate):
        pass
    """
    @abc.abstractmethod
    def get_config(self):
        pass
