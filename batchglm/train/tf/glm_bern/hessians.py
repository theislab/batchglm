import tensorflow as tf

import logging

from .external import HessianGLMALL

logger = logging.getLogger(__name__)


class Hessians(HessianGLMALL):

    def _weight_hessian_ab(
            self,
            X,
            loc,
            scale,
    ):
        return tf.zeros_like(loc)

    def _weight_hessian_aa(
            self,
            X,
            loc,
            scale,
    ):
        const = - loc * (1-loc)
        return const

    def _weight_hessian_bb(
            self,
            X,
            loc,
            scale,
    ):
        return tf.zeros_like(loc)


