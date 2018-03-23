import tensorflow as tf
import numpy as np

import tensorflow.contrib.eager as tfe

# tfe.enable_eager_execution()

from impl.tf.base import TFSession
from models.negative_binomial.linear_bias import Simulator

sim = Simulator()
# sim.generate()
# sim.save("resources/")

sim.load("resources/")

from impl.tf.negative_binomial.linear_bias import AbstractEstimator, TFEstimator, \
    TFEstimatorGraph, fit_partitioned_nb

design = sim.data.design
data = sim.data.sample_data

(r, p) = fit_partitioned_nb(data, design.astype(np.int32))

sess = tf.InteractiveSession()

(real_r, real_p) = sess.run((r, p))
print(sim.r[:, 0])
print(real_r[:, 0])

solve = tf.matrix_solve_ls(design, tf.log(r))
estim_params = sess.run(solve)
print(estim_params)
print(sim.params["bias_r"])

# x = tf.constant(design, name="X")
# y = r[:, 0]
#
# xt= tf.transpose(x, name="Xt")
# xtx = tf.matmul(xt, x, name="XtX")
# xtx_inv = tf.matrix_inverse(xtx, name="XtX_inv")
# xty = tf.matmul(xt, tf.expand_dims(y, 1), name="XtY")
