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
        scalar_one = tf.constant(1, shape=(), dtype=self.dtype)
        const = loc * (tf.digamma(loc + scale) - tf.digamma(loc) + loc * (tf.polygamma(scalar_one, loc + scale) - tf.polygamma(scalar_one, loc)))
        const2 = const + loc * loc / (loc + scale)

        return const2

    def _weight_fim_bb(
            self,
            loc,
            scale
    ):
        scalar_one = tf.constant(1, shape=(), dtype=self.dtype)
        const = scale * (tf.digamma(loc + scale) - tf.digamma(scale) + scale * (
                    tf.polygamma(scalar_one, loc + scale) - tf.polygamma(scalar_one, scale)))
        const2 = const + scale * scale / (loc + scale)

        return const2

    def _weight_fim_ab(
            self,
            loc,
            scale
    ):
        scalar_one = tf.constant(1, shape=(), dtype=self.dtype)
        const = tf.polygamma(scalar_one, loc + scale) * loc * scale

        return const