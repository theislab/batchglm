import tensorflow as tf

import logging

from .external import FIMGLMALL

logger = logging.getLogger(__name__)


class FIM(FIMGLMALL):
    # No Fisher Information Matrices due to unsolvable E[log(X)]

    def _weight_fim_aa(
            self,
            loc,
            scale
    ):
        assert False, "not implemented"

    def _weight_fim_bb(
            self,
            loc,
            scale
    ):
        assert False, "not implemented"
