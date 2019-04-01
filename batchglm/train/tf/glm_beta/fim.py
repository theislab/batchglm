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
        return 0

    def _weight_fim_bb(
            self,
            loc,
            scale
    ):

        return 0

    def _weight_fim(
            self,
            loc,
            scale
    ):
        scalar_one = tf.constant(1, shape=(), dtype=self.dtype)

        # aa:
        const1 = loc * (tf.digamma(loc + scale) - tf.digamma(loc) + loc * (
                    tf.polygamma(scalar_one, loc + scale) - tf.polygamma(scalar_one, loc)))
        aa_part = const1 + loc * loc / (loc + scale)

        # bb:
        const2 = scale * (tf.digamma(loc + scale) - tf.digamma(scale) + scale * (
                tf.polygamma(scalar_one, loc + scale) - tf.polygamma(scalar_one, scale)))
        bb_part = const2 + scale * scale / (loc + scale)

        # ab
        ab_part = tf.polygamma(scalar_one, loc + scale) * loc * scale

        # should be 4 dimensional object, first two dimensions are dimensions of loc/scale, third and forth should be
        # the dimensions of the [[aa, ab], [ab, bb]] matrices per element of loc/scale
        # (aa, ab, bb scalars)
        # not tested yet!
        full_fim = tf.stack([tf.stack([aa_part, ab_part], axis=2), tf.stack([ab_part, bb_part], axis=2)], axis=3)

        return full_fim