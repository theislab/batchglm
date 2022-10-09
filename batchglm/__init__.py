import os

__author__ = "Mario Picciani"
__email__ = "mario.picciani@tum.de"
__version__ = "0.7.4"
from . import models, pkg_constants, train, utils
from .log_cfg import logger, setup_logging, unconfigure_logging

# we need this for the sparse package, see https://github.com/pydata/sparse/issues/10
os.environ["SPARSE_AUTO_DENSIFY"] = "1"
