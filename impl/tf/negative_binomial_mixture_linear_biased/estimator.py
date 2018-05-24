import abc
from typing import Union

import os
import datetime

import xarray as xr
import tensorflow as tf
import numpy as np

import impl.tf.util as tf_utils
from .external import AbstractEstimator, TFEstimator, TFEstimatorGraph
from .external import nb_utils, tf_linreg
from tensorflow.contrib.distributions import Distribution


class MixtureModel:
    initializer: Union[tf.Tensor, tf.Operation]
    prob: tf.Tensor
    logit_prob: tf.Tensor
    
    def __init__(self, initial_values, name="mixture_prob"):
        with tf.name_scope(name):
            # optimize logits to keep `mixture_prob` between the interval [0, 1]
            logit_prob = tf.Variable(tf_utils.logit(initial_values), name="logit_prob")
            prob = tf.sigmoid(logit_prob.initialized_value(), name="prob")
            
            # normalize: the assignment probabilities should sum up to 1
            # => `sum(mixture_prob of one sample) = 1`
            with tf.name_scope("normalize"):
                prob = prob / tf.reduce_sum(prob, axis=0, keepdims=True)
            
            self.initializer = logit_prob.initializer
            self.prob = prob
            self.log_prob = tf.log(prob)
            self.logit_prob = logit_prob
        # assert (mixture_prob.shape == (num_mixtures, num_samples, 1))


class LinearBatchModel:
    a: tf.Tensor
    a_intercept: tf.Variable
    a_slope: tf.Variable
    b: tf.Tensor
    b_intercept: tf.Variable
    b_slope: tf.Variable
    log_mu_obs: tf.Tensor
    log_r_obs: tf.Tensor
    log_count_probs: tf.Tensor
    joint_log_probs: tf.Tensor
    loss: tf.Tensor
    
    def __init__(self,
                 init_dist: nb_utils.NegativeBinomial,
                 sample_index,
                 sample_data,
                 design,
                 mixture_model: MixtureModel,
                 name="Linear_Batch_Model"):
        with tf.name_scope(name):
            num_mixtures = mixture_model.prob.shape[0]
            num_design_params = design.shape[-1]
            (batch_size, num_genes) = sample_data.shape
            
            init_a_intercept = tf.log(init_dist.mean())
            init_b_intercept = tf.log(init_dist.r)
            
            mixture_log_prob = tf.gather(mixture_model.log_prob, sample_index, axis=-1)
            
            assert sample_data.shape == [batch_size, num_genes]
            assert design.shape == [batch_size, num_design_params]
            assert mixture_log_prob.shape == [num_mixtures, batch_size]
            assert init_a_intercept.shape == [num_mixtures, 1, num_genes] == init_b_intercept.shape
            
            init_a_slopes = tf.truncated_normal(tf.TensorShape([1, num_design_params - 1, num_genes]),
                                                dtype=design.dtype)
            
            init_b_slopes = init_a_slopes
            
            a, a_intercept, a_slope = tf_linreg.param_variable(init_a_intercept, init_a_slopes, name="a")
            b, b_intercept, b_slope = tf_linreg.param_variable(init_b_intercept, init_b_slopes, name="b")
            assert a.shape == (num_mixtures, num_design_params, num_genes) == b.shape
            
            dist_estim = nb_utils.NegativeBinomial(mean=tf.exp(a_intercept),
                                                   r=tf.exp(b_intercept),
                                                   name="dist_estim")
            
            with tf.name_scope("broadcast"):
                design = tf.expand_dims(design, axis=0)
            design = tf.tile(design, (num_mixtures, 1, 1))
            assert (design.shape == (num_mixtures, batch_size, num_design_params))
            
            with tf.name_scope("mu"):
                log_mu = tf.matmul(design, a, name="log_mu_obs")
                log_mu = tf.clip_by_value(log_mu, log_mu.dtype.min, log_mu.dtype.max)
                mu = tf.exp(log_mu)
            
            with tf.name_scope("r"):
                log_r = tf.matmul(design, b, name="log_r_obs")
                log_r = tf.clip_by_value(log_r, log_r.dtype.min, log_r.dtype.max)
                r = tf.exp(log_r)
            
            dist_obs = nb_utils.NegativeBinomial(r=r, mean=mu, name="dist_obs")
            
            # calculate mixture model probability:
            log_count_probs = dist_obs.log_prob(tf.expand_dims(sample_data, 0), name="log_count_probs")
            
            # sum up: for k in num_mixtures: mixture_prob(k) * P(r_k, mu_k, sample_data)
            joint_log_probs = tf.reduce_logsumexp(log_count_probs + tf.expand_dims(mixture_log_prob, -1),
                                                  axis=-3,
                                                  name="joint_log_probs")
            
            # probs = tf.exp(joint_log_probs, name="joint_probs")
            
            # minimize negative log probability (log(1) = 0);
            # use the mean loss to keep a constant learning rate independently of the batch size
            loss = -tf.reduce_mean(joint_log_probs, name="loss")
            
            self.a = a
            self.a_intercept = a_intercept
            self.a_slope = a_slope
            self.b = b
            self.b_intercept = b_intercept
            self.b_slope = b_slope
            self.dist_estim = dist_estim
            self.dist_obs = dist_obs
            self.log_mu_obs = log_mu
            self.log_r_obs = log_r
            self.log_count_probs = log_count_probs
            self.joint_log_probs = joint_log_probs
            self.loss = loss


class EstimatorGraph(TFEstimatorGraph):
    sample_data: tf.Tensor
    design: tf.Tensor
    
    dist_obs: Distribution
    dist_estim: Distribution
    
    mu: tf.Tensor
    sigma2: tf.Tensor
    a: tf.Tensor
    b: tf.Tensor
    mixture_prob: tf.Tensor
    mixture_assignment: tf.Tensor
    
    def __init__(
            self,
            num_mixtures,
            num_samples,
            num_genes,
            num_design_params,
            batch_size=250,
            graph=None,
            random_effect=0.1,
            log_dir=None
    ):
        super().__init__(graph)
        if log_dir is None:
            log_dir = os.path.join("data/log/", self.__module__, datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
        self.log_dir = log_dir
        
        # initial graph elements
        with self.graph.as_default():
            # placeholders
            sample_data = tf_utils.caching_placeholder(tf.float32, shape=(num_samples, num_genes), name="sample_data")
            design = tf_utils.caching_placeholder(tf.float32, shape=(num_samples, num_design_params), name="design")
            initial_mixture_probs = tf_utils.caching_placeholder(tf.float32,
                                                                 shape=(num_mixtures, num_samples),
                                                                 name="initial_mixture_probs")
            
            learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
            # train_steps = tf.placeholder(tf.int32, shape=(), name="training_steps")
            
            # apply a random intercept to avoid zero gradients and infinite values
            with tf.name_scope("randomize"):
                initial_mixture_probs += tf.random_uniform(initial_mixture_probs.shape, 0, random_effect,
                                                           dtype=tf.float32)
                initial_mixture_probs = initial_mixture_probs / tf.reduce_sum(initial_mixture_probs, axis=0,
                                                                              keepdims=True)
            
            with tf.name_scope("broadcast"):
                design_bcast = tf.expand_dims(design, axis=0)
                design_bcast = tf.tile(design_bcast, (num_mixtures, 1, 1))
                assert (design_bcast.shape == (num_mixtures, num_samples, num_design_params))
            
            with tf.name_scope("initialization"):
                # implicit broadcasting of sample_data and initial_mixture_probs to
                #   shape (num_mixtures, num_samples, num_genes)
                init_dist = nb_utils.fit(tf.expand_dims(sample_data, 0),
                                         weights=tf.expand_dims(initial_mixture_probs, -1), axis=-2)
                assert init_dist.r.shape == [num_mixtures, 1, num_genes]
            
            # define mixture parameters
            mixture_model = MixtureModel(initial_mixture_probs)
            
            data = tf.data.Dataset.from_tensor_slices((
                tf.range(num_samples, name="sample_index"),
                sample_data,
                design
            ))
            data = data.repeat()
            data = data.shuffle(batch_size * 5)
            data = data.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
            
            iterator = data.make_initializable_iterator()
            batch_sample_index, batch_sample_data, batch_design = iterator.get_next()
            
            # Batch model:
            #     only `batch_size` samples will be used;
            #     All per-sample variables have to be passed via `data`.
            #     Sample-independent variables (e.g. per-gene distributions) can be created inside the batch model
            batch_model = LinearBatchModel(
                init_dist,
                batch_sample_index,
                batch_sample_data,
                batch_design,
                mixture_model
            )
            
            with tf.name_scope("mu_estim"):
                mu_estim = tf.exp(tf.tile(batch_model.a_intercept, (1, num_samples, 1)))
            with tf.name_scope("r_estim"):
                r_estim = tf.exp(tf.tile(batch_model.b_intercept, (1, num_samples, 1)))
            dist_estim = nb_utils.NegativeBinomial(mean=mu_estim, r=r_estim)
            
            with tf.name_scope("mu_obs"):
                mu_obs = tf.exp(tf.matmul(design_bcast, batch_model.a))
            with tf.name_scope("r_obs"):
                r_obs = tf.exp(tf.matmul(design_bcast, batch_model.b))
            dist_obs = nb_utils.NegativeBinomial(mean=mu_obs, r=r_obs)
            
            with tf.name_scope('summaries'):
                tf.summary.histogram('a_intercept', batch_model.a_intercept)
                tf.summary.histogram('b_intercept', batch_model.b_intercept)
                tf.summary.histogram('a_slope', batch_model.a_slope)
                tf.summary.histogram('b_slope', batch_model.b_slope)
                tf.summary.scalar('loss', batch_model.loss)
                
                with tf.name_scope("prob_image"):
                    # repeat:
                    prob_image = tf.reshape(
                        tf.transpose(tf.tile(
                            [mixture_model.prob],  # input tensor
                            ((num_samples // num_mixtures), 1, 1))),  # target shape
                        [-1]  # flatten
                    )
                    prob_image = tf.transpose(
                        tf.reshape(prob_image, ((num_samples // num_mixtures) * num_mixtures, num_samples)))
                    prob_image = tf.expand_dims(prob_image, 0)
                    prob_image = tf.expand_dims(prob_image, -1)
                    prob_image = prob_image * 255.0
                
                tf.summary.image('mixture_prob', prob_image)
            
            # set up class attributes
            self.sample_data = sample_data
            self.design = design
            
            self.distribution_estim = dist_estim
            self.distribution_obs = dist_obs
            self.batch_model = batch_model
            
            self.mu = tf.reduce_sum(dist_obs.mean() * tf.expand_dims(mixture_model.prob, axis=-1), axis=-3)
            self.sigma2 = tf.reduce_sum(dist_obs.variance() * tf.expand_dims(mixture_model.prob, axis=-1), axis=-3)
            self.a = batch_model.a
            self.b = batch_model.b
            assert (self.mu.shape == (num_samples, num_genes))
            assert (self.sigma2.shape == (num_samples, num_genes))
            assert (self.a.shape == (num_mixtures, num_design_params, num_genes))
            assert (self.b.shape == (num_mixtures, num_design_params, num_genes))
            
            self.mixture_prob = mixture_model.prob
            with tf.name_scope("mixture_assignment"):
                self.mixture_assignment = tf.argmax(mixture_model.prob, axis=0)
            assert (self.mixture_prob.shape == (num_mixtures, num_samples))
            assert (self.mixture_assignment.shape == num_samples)
            
            # training
            self.loss = batch_model.loss
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.gradient = self.optimizer.compute_gradients(self.loss)
            
            self.global_train_step = tf.Variable(0, name="global_step")
            
            # def minimizer(_):
            #     with tf.control_dependencies([
            #         iterstepper.step_op(),
            #         tf.Print(batch_sample_index, [self.loss], "loss: ")
            #     ]):
            #         retval = self.optimizer.minimize(self.loss, global_step=self.global_train_step)
            #     return retval
            #
            # self.train_op = tf_utils.for_loop(
            #     condition=lambda i: tf.less(i, train_steps),
            #     modifier=lambda i: tf.add(i, 1),
            #     body_op=minimizer
            # )
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_train_step)
            
            self.initializer_ops = [
                tf.variables_initializer([sample_data, design]),
                iterator.initializer,
                tf.variables_initializer(tf.global_variables()),
            ]
            
            self.saver = tf.train.Saver()
            
            self.merged_summary = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(log_dir, self.graph)
    
    def initialize(self, session, feed_dict, **kwargs):
        with self.graph.as_default():
            for op in self.initializer_ops:
                session.run(op, feed_dict=feed_dict)
    
    def train(self, session, feed_dict, *args, steps=1000, learning_rate=0.05, **kwargs):
        print("learning rate: %s" % learning_rate)
        
        # feed_dict = feed_dict.copy()
        feed_dict = dict()
        feed_dict["learning_rate:0"] = learning_rate
        
        loss_res = None
        for i in range(steps):
            (train_step, loss_res, _) = session.run((self.global_train_step, self.loss, self.train_op),
                                                    feed_dict=feed_dict)
            
            if train_step % 10 == 0:  # Record summaries and test-set accuracy
                summary, = session.run([self.merged_summary])
                self.summary_writer.add_summary(summary, train_step)
                self.save(session)
            
            print("Step: %d\tloss: %f" % (train_step, loss_res))
            
            if np.isnan(loss_res) or np.isinf(loss_res):
                print("WARNING: invalid loss!")
                return loss_res
        
        return loss_res
    
    def save(self, session):
        self.saver.save(session, os.path.join(self.log_dir, "state"), global_step=self.global_train_step)
    
    def restore(self, session, state_file):
        self.saver.restore(session, state_file)


# g = EstimatorGraph(sim.data.design, optimizable_nb=False)
# writer = tf.summary.FileWriter("/tmp/log/...", g.graph)


class Estimator(AbstractEstimator, TFEstimator, metaclass=abc.ABCMeta):
    model: EstimatorGraph
    
    def __init__(self, input_data: xr.Dataset,
                 batch_size=250,
                 model=None):
        if model is None:
            num_mixtures = input_data.dims["mixtures"]
            num_genes = input_data.dims["genes"]
            num_samples = input_data.dims["samples"]
            num_design_params = input_data.dims["design_params"]
            
            tf.reset_default_graph()
            
            model = EstimatorGraph(num_mixtures, num_samples, num_genes, num_design_params,
                                   batch_size=batch_size,
                                   graph=tf.get_default_graph())
        
        TFEstimator.__init__(self, input_data, model)
    
    @property
    def loss(self):
        return self.run(self.model.loss)
    
    @property
    def mu(self):
        return self.run(self.model.mu)
    
    @property
    def sigma2(self):
        return self.run(self.model.sigma2)
    
    @property
    def a(self):
        return self.run(self.model.a)
    
    @property
    def b(self):
        return self.run(self.model.b)
    
    @property
    def mixture_assignment(self):
        return self.run(self.model.mixture_assignment)
    
    @property
    def mixture_prob(self):
        return self.run(self.model.mixture_prob)
