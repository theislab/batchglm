import tensorflow as tf

import logging

from .external import FIMGLMALL

logger = logging.getLogger(__name__)


class FIM(FIMGLMALL):

    def _weight_fim_aa(
            self,
            loc,
            scale
    ):
        const = - loc * (1-loc)

        return const

    def _weight_fim_bb(
            self,
            loc,
            scale
    ):
        return tf.zeros_like(loc)
