# absolute imports for modules

# relative imports
from .base import Model, InputData
from .base import Model as NegativeBinomialWithLinearBiasModel  # Alias for Model
from .base import InputData as NegativeBinomialWithLinearBiasInputData  # Alias for InputData

from .simulator import Simulator
from .simulator import Simulator as NegativeBinomialWithLinearBiasSimulator  # Alias for Simulator

# use TF as default estimator implementation
from impl.tf.negative_binomial_linear_biased import Estimator as NegativeBinomialWithLinearBiasEstimator

__all__ = ['NegativeBinomialWithLinearBiasSimulator',
           'NegativeBinomialWithLinearBiasInputData',
           'NegativeBinomialWithLinearBiasModel',
           'NegativeBinomialWithLinearBiasEstimator']
