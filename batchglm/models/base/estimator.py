import abc
from enum import Enum
import logging

try:
    import anndata
except ImportError:
    anndata = None

from .model import _Model_Base

logger = logging.getLogger(__name__)


class _Estimator_Base(_Model_Base, metaclass=abc.ABCMeta):
    r"""
    Estimator base class
    """

    class TrainingStrategy(Enum):
        AUTO = None

    def validate_data(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def initialize(self, **kwargs):
        """
        Initializes this estimator
        """
        pass

    @abc.abstractmethod
    def train(self, learning_rate=None, **kwargs):
        """
        Starts the training routine
        """
        pass

    def train_sequence(self, training_strategy=TrainingStrategy.AUTO):
        """
        Starts a sequence of training routines

        :param training_strategy: List of dicts or enum with parameters which will be passed to self.train().

                - `training_strategy = [ {"learning_rate": 0.5}, {"learning_rate": 0.05} ]` is equivalent to
                    `self.train(learning_rate=0.5); self.train(learning_rate=0.05);`

                - Can also be an enum: self.TrainingStrategy.[AUTO|DEFAULT|EXACT|QUICK|...]
                - Can also be a str: "[AUTO|DEFAULT|EXACT|QUICK|...]"
        """
        if isinstance(training_strategy, Enum):
            training_strategy = training_strategy.value
        elif isinstance(training_strategy, str):
            training_strategy = self.TrainingStrategy[training_strategy].value

        if training_strategy is None:
            training_strategy = self.TrainingStrategy.DEFAULT.value

        logger.info("training strategy: %s", str(training_strategy))

        for idx, d in enumerate(training_strategy):
            logger.info("Beginning with training sequence #%d", idx + 1)
            self.train(**d)
            logger.info("Training sequence #%d complete", idx + 1)

    @abc.abstractmethod
    def finalize(self, **kwargs):
        """
        Clean up, free resources

        :return: some Estimator containing all necessary data
        """
        pass

    @property
    @abc.abstractmethod
    def loss(self):
        pass

    @property
    @abc.abstractmethod
    def gradients(self):
        pass


class _EstimatorStore_XArray_Base():

    def __init__(self):
        pass

    def initialize(self, **kwargs):
        raise NotImplementedError("This object only stores estimated values")

    def train(self, **kwargs):
        raise NotImplementedError("This object only stores estimated values")

    def finalize(self, **kwargs):
        return self

    def validate_data(self, **kwargs):
        raise NotImplementedError("This object only stores estimated values")

    @property
    def input_data(self):
        return self._input_data

    @property
    def X(self):
        return self.input_data.X

    @property
    def features(self):
        return self.input_data.features

    @property
    def loss(self):
        return self.params["loss"]
