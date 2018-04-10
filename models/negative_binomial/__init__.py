from .base import Model, InputData, AbstractEstimator
from .base import Model as NegativeBinomialModel  # Alias for Model
from .base import InputData as NegativeBinomialInputData  # Alias for InputData

from .simulator import Simulator
from .simulator import Simulator as NegativeBinomialSimulator  # Alias for Simulator

# from .estimator import Estimator
# from .estimator import Estimator as NegativeBinomialEstimator  # Alias for Estimator

__all__ = [
    'Simulator',
    'NegativeBinomialSimulator',
    'NegativeBinomialInputData',
    'NegativeBinomialModel',
    'AbstractEstimator'
    # 'Estimator'
    # 'NegativeBinomialEstimator'
]
