import os

from . import models, train, utils
from ._version import get_versions
from .log_cfg import logger, setup_logging, unconfigure_logging

__version__ = get_versions()["version"]
del get_versions

# we need this for the sparse package, see https://github.com/pydata/sparse/issues/10
os.environ["SPARSE_AUTO_DENSIFY"] = "1"
