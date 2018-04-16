import abc

import tensorflow as tf
import numpy as np

from .external import AbstractEstimator, TFEstimator, TFEstimatorGraph
from .external import nb_utils, LinearRegression


class EstimatorGraph(TFEstimatorGraph):
    sample_data: tf.Tensor
    
    r: tf.Tensor
    p: tf.Tensor
    mu: tf.Tensor
    a: tf.Tensor
    b: tf.Tensor
    
    def __init__(
            self,
            num_samples,
            num_genes,
            graph=None,
            optimizable_nb=False,
            optimizable_lm=False
    ):
        super().__init__(graph)
        
        # initial graph elements
        with self.graph.as_default():
            sample_data = tf.placeholder(tf.float32, shape=(num_samples, num_genes), name="sample_data")
            design = tf.placeholder(tf.float32, shape=(num_samples, None), name="design")
            
            ### waiting for tf.unique_with_counts_v2 to be released:
            # _, partition_index = tf.unique(design, axis=0, name="unique")
            ### alternatively use python:
            with tf.name_scope("partitions"):
                _, partitions = tf.py_func(
                    lambda x: np.unique(x, axis=0, return_inverse=True),
                    [design],
                    [design.dtype, tf.int64]
                )
                partitions = tf.cast(partitions, tf.int32)
            
            dist = nb_utils.fit_partitioned(sample_data, partitions, optimizable=optimizable_nb,
                                            name="background_NB-dist")
            r = dist.r
            p = dist.p
            mu = dist.mu
            log_r = tf.log(r, name="log_r")
            log_p = tf.log(p, name="log_p")
            log_mu = tf.log(mu, name="log_mu")
            
            linreg_a = LinearRegression(design, log_r, name="lin_reg_a", fast=not optimizable_lm)
            a = linreg_a.estimated_params
            a = tf.identity(a, "a")
            
            linreg_b = LinearRegression(design, log_mu, name="lin_reg_b", fast=not optimizable_lm)
            b = linreg_b.estimated_params
            b = tf.identity(b, "b")
            
            r_true = tf.exp(tf.gather(a, 0), name="true_r")
            mu_true = tf.exp(tf.gather(b, 0), name="true_mu")
            
            distribution = nb_utils.NegativeBinomial(r=r_true, mu=mu_true, name="true_NB-dist")
            log_probs = tf.identity(distribution.log_prob(sample_data), name="log_probs")
            
            with tf.name_scope("training"):
                # minimize negative log probability (log(1) = 0)
                loss = -tf.reduce_sum(log_probs, name="loss")
                
                # define train function
                train_op = None
                if optimizable_nb:
                    optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
                    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            
            self.sample_data = sample_data
            self.design_tensor = design
            self.design = design
            
            self.initializer_op = tf.global_variables_initializer()
            self.r = r
            self.p = p
            self.mu = mu
            self.log_r = log_r
            self.log_p = log_p
            self.log_mu = log_mu
            
            self.a = a
            self.b = b
            self.r_true = r_true
            self.mu_true = mu_true
            
            self.distribution = distribution
            self.log_probs = log_probs
            
            self.loss = loss
            self.train_op = train_op
    
    def initialize(self, session, feed_dict, **kwargs):
        session.run(self.initializer_op, feed_dict=feed_dict)
    
    def train(self, session, feed_dict, *args, steps=1000, **kwargs):
        if self.train_op is None:
            raise RuntimeWarning("this graph is not trainable")
        errors = []
        for i in range(steps):
            (loss_res, _) = session.run((self.loss, self.train_op),
                                        feed_dict=feed_dict)
            errors.append(loss_res)
            print(i)
        
        return errors


# g = EstimatorGraph(sim.data.design, optimizable_nb=False)
# writer = tf.summary.FileWriter("/tmp/log/...", g.graph)


class Estimator(AbstractEstimator, TFEstimator, metaclass=abc.ABCMeta):
    model: EstimatorGraph
    
    def __init__(self, input_data: dict, tf_estimator_graph=None):
        if tf_estimator_graph is None:
            (num_samples, num_genes) = input_data["sample_data"].shape
            tf_estimator_graph = EstimatorGraph(num_samples, num_genes)
        
        TFEstimator.__init__(self, input_data, tf_estimator_graph)
    
    @property
    def loss(self):
        return self.run(self.model.loss)
    
    @property
    def r(self):
        return self.run(self.model.r)
    
    @property
    def p(self):
        return self.run(self.model.p)
    
    @property
    def mu(self):
        return self.run(self.model.mu)
    
    @property
    def a(self):
        return self.run(self.model.a)
    
    @property
    def b(self):
        return self.run(self.model.b)
