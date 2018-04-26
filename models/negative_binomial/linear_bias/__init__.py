# absolute imports for modules
from models.negative_binomial import NegativeBinomialModel, NegativeBinomialInputData, NegativeBinomialSimulator

# relative imports
from .base import *
from .base import Model as NegativeBinomialWithLinearBiasModel  # Alias for Model
from .base import InputData as NegativeBinomialWithLinearBiasInputData  # Alias for InputData

from .simulator import *
from .simulator import Simulator as NegativeBinomialWithLinearBiasSimulator  # Alias for Simulator

# use TF as default estimator implementation
from impl.tf.negative_binomial.linear_bias.estimator import Estimator
from impl.tf.negative_binomial.linear_bias.estimator import Estimator as NegativeBinomialWithLinearBiasEstimator

__all__ = ['NegativeBinomialWithLinearBiasSimulator',
           'NegativeBinomialWithLinearBiasInputData',
           'NegativeBinomialWithLinearBiasModel',
           'NegativeBinomialWithLinearBiasEstimator']
