from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

import os

from .log_cfg import logger, setup_logging, unconfigure_logging

# we need this for the sparse package, see https://github.com/pydata/sparse/issues/10
os.environ["SPARSE_AUTO_DENSIFY"] = "1"
