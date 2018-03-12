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


(r, p) = fit_partitioned_nb(data, design)

sess = tf.InteractiveSession()

(real_r, real_p) = sess.run((r, p))