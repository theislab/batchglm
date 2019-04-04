import logging

from .external import ReducableTensorsGLMALL
from .hessians import Hessians
from .jacobians import Jacobians
from .fim import FIM

logger = logging.getLogger("batchglm")


class ReducibleTensors(Jacobians, Hessians, FIM, ReducableTensorsGLMALL):
    """
    """
