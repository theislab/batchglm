from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

from .log_cfg import logger, unconfigure_logging, setup_logging

import os
# we need this for the sparse package, see https://github.com/pydata/sparse/issues/10
os.environ["SPARSE_AUTO_DENSIFY"] = "1"
