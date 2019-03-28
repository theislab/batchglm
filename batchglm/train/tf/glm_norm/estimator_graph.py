import logging

from .model import ProcessModel
from .external import EstimatorGraphAll

logger = logging.getLogger(__name__)


class EstimatorGraph(ProcessModel, EstimatorGraphAll):
    """
    Full class.
    """
