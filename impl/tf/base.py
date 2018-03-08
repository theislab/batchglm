import tensorflow as tf
import numpy as np


class BasicEstimatorGraph:

class BasicSession:
    
    def __init__(self, tf_model):
        self.model = tf_model # type: RSA_negative_binomial
        self.create_new_session(sample_data)
    
    def create_new_session(self, sample_data):
        with self.model.graph.as_default():
            self.session = tf.Session()
            self.feed_dict = {
                self.model.sample_data: sample_data
            }
            return self.session
    
    def initialize(self):
        self.model.initialize(self.session, self.feed_dict)
    
    def train(self, steps):
        self.model.train(steps, self.session, self.feed_dict)
    
    # TODO: hässlich; dämliches Maß; nur für einen Param
    def compare(self, session, feed_dict, real_values):
        print(np.nanmean(
            np.abs(self.r - np.array(real_values.r)) /
            np.fmax(self.r, np.array(real_values.r))
        ))
    
    def evaluate(self, s):
        pass
    
    @property
    def loss(self):
        return self.session.run(self.model.loss, feed_dict=self.feed_dict)
    
    @property
    def r(self):
        return self.session.run(self.model.r, feed_dict=self.feed_dict)
    
    @property
    def p(self):
        return self.session.run(self.model.p, feed_dict=self.feed_dict)
    
    @property
    def mu(self):
        return self.session.run(self.model.mu, feed_dict=self.feed_dict)