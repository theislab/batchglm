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


def as_design_matrix(design, num_bits=8):
    def uint2bits(int_ar_in, num_bits):
        """ convert (numpyarray of uint => array of Nbits bits) for many bits in parallel"""
        in_size__t = int_ar_in.shape
        in_flat = int_ar_in.flatten()
        out_num_bit = np.zeros((len(in_flat), num_bits))
        for i_bits in range(num_bits):
            out_num_bit[:, i_bits] = (in_flat >> i_bits) & 1
        out_num_bit = out_num_bit.reshape(in_size__t + (num_bits,))
        return out_num_bit

    # design_bin = np.unpackbits(np.expand_dims(design, 1).astype(np.uint8), axis=1)
    design_bin = uint2bits(design.astype(np.uint32), 8)
    mask = np.all(design_bin == 0, axis=0)
    design_bin = design_bin[:, ~mask]
    # design_bin = np.unique(design_bin, axis=0)
    # design_bin = np.unique(design_bin, axis=1)

    # add bias column
    n, m = design_bin.shape  # for generality
    ones = np.ones((n, 1))
    design_bin = np.hstack((ones, design_bin))

    return design_bin


design_bin = as_design_matrix(design)
np.unique(design_bin, axis=0)

tf.matrix_solve_ls

from patsy import dmatrix
