import tensorflow as tf
import numpy as np

from tensorflow.contrib import eager as tfe

tfe.enable_eager_execution()

import impl.tf.negative_binomial.util as nb_utils
from models.negative_binomial_mixture import Simulator

sim = Simulator()
sim.generate()
# sim.save("resources/")
# sim.load("resources/")

sample_data = sim.data.sample_data
initial_mixture_probs = sim.data.initial_mixture_probs

sample_data = tf.tile(sample_data, (initial_mixture_probs.shape[0], 1, 1))
(num_mixtures, num_samples, num_distributions) = tf.shape(sample_data)

mixture_prob = initial_mixture_probs

comp_data = tf.gather(sample_data[0], tf.squeeze(tf.where(initial_mixture_probs[0] == 1)))

comp_dist = nb_utils.fit(sample_data=comp_data,
                         # weights=1,
                         name="fit_nb-dist")

distribution = nb_utils.fit(sample_data=sample_data,
                            axis=-2,
                            weights=tf.expand_dims(mixture_prob, axis=-1),
                            name="fit_nb-dist")

assert np.all(np.asarray(distribution.r[0]) == np.asarray(comp_dist.r))
