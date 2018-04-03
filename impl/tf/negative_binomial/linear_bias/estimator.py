import abc

import tensorflow as tf

from . import AbstractEstimator, TFEstimator, TFEstimatorGraph, fit_partitioned_nb, linear_regression, NegativeBinomial


class EstimatorGraph(TFEstimatorGraph):
    sample_data: tf.Tensor
    
    r: tf.Tensor
    p: tf.Tensor
    mu: tf.Tensor
    a: tf.Tensor
    b: tf.Tensor
    
    def __init__(self, design, graph=tf.Graph(), optimizable_nb=False, optimizable_lm=False):
        super().__init__(graph)
        
        # initial graph elements
        with self.graph.as_default():
            sample_data = tf.placeholder(tf.float32, name="sample_data")
            design_tensor = tf.constant(design, dtype=tf.int32, name="design")
            
            dist = fit_partitioned_nb(sample_data, design, optimizable=optimizable_nb,
                                      name="background_NB-dist")
            r = dist.r
            p = dist.p
            mu = dist.mu
            log_r = tf.log(r, name="log_r")
            log_p = tf.log(p, name="log_p")
            log_mu = tf.log(mu, name="log_mu")
            
            design_tensor_asfloat = tf.cast(design_tensor, dtype=r.dtype, name="design_casted")
            a, loss = linear_regression(design_tensor_asfloat, log_r, name="lin_reg_a",
                                        fast=not optimizable_lm)
            tf.identity(a, "a")
            b, loss = linear_regression(design_tensor_asfloat, log_mu, name="lin_reg_b",
                                        fast=not optimizable_lm)
            tf.identity(b, "b")
            
            bias_r = tf.exp(a, name="bias_r")
            bias_mu = tf.exp(b, name="bias_mu")
            
            r_true = tf.gather(a, 0, name="true_r")
            mu_true = tf.gather(b, 0, name="true_mu")
            
            distribution = NegativeBinomial(r=r_true, mu=mu_true, name="true_NB-dist")
            log_probs = tf.identity(distribution.log_prob(sample_data), name="log_probs")
            
            with tf.name_scope("training"):
                # minimize negative log probability (log(1) = 0)
                loss = -tf.reduce_sum(log_probs, name="loss")
                
                # define train function
                train_op = None
                if optimizable_nb:
                    train_op = tf.train.AdamOptimizer(learning_rate=0.05)
                    train_op = train_op.minimize(loss, global_step=tf.train.get_global_step())
            
            self.sample_data = sample_data
            self.design_tensor = design_tensor
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
            self.log_r_true = bias_r
            self.log_mu_true = bias_mu
            
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
            tf_estimator_graph = EstimatorGraph(input_data["design"])
        
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
    def bias_mu(self):
        return self.run(self.model.log_mu_true)
    
    @property
    def bias_r(self):
        return self.run(self.model.log_r_true)
