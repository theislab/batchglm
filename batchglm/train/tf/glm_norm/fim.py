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
        W = tf.square(tf.divide(tf.ones_like(scale), scale))

        return W

    def _weight_fim_bb(
            self,
            loc,
            scale
    ):
        W = tf.constant(2, shape=loc.shape, dtype=self.dtype)

        return W
