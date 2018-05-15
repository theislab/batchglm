import abc
from typing import Tuple

import tensorflow as tf
import numpy as np

import impl.tf.util as tf_utils
from .external import AbstractEstimator, TFEstimator, TFEstimatorGraph
from .external import nb_utils, tf_linreg


class LinearBatchModel:
    a: tf.Tensor
    a_intercept: tf.Variable
    a_slope: tf.Variable
    b: tf.Tensor
    b_intercept: tf.Variable
    b_slope: tf.Variable
    dist_estim: nb_utils.NegativeBinomial
    dist_obs: nb_utils.NegativeBinomial
    log_mu_obs: tf.Tensor
    log_r_obs: tf.Tensor
    log_count_probs: tf.Tensor
    joint_log_probs: tf.Tensor
    loss: tf.Tensor

    def __init__(self,
                 init_dist: nb_utils.NegativeBinomial,
                 sample_data,
                 design,
                 name="Linear_Batch_Model"):
        with tf.name_scope(name):
            num_design_params = design.shape[-1]
            (batch_size, num_genes) = sample_data.shape

            init_a_intercept = tf.log(init_dist.mean())
            init_b_intercept = tf.log(init_dist.r)

            assert sample_data.shape == [batch_size, num_genes]
            assert design.shape == [batch_size, num_design_params]
            assert init_a_intercept.shape == [1, num_genes] == init_b_intercept.shape

            init_a_slopes = tf.abs(tf.truncated_normal(tf.TensorShape([num_design_params - 1, num_genes]),
                                                       dtype=design.dtype))

            init_b_slopes = init_a_slopes

            a, a_intercept, a_slope = tf_linreg.param_variable(init_a_intercept, init_a_slopes, name="a")
            b, b_intercept, b_slope = tf_linreg.param_variable(init_b_intercept, init_b_slopes, name="b")
            assert a.shape == (num_design_params, num_genes) == b.shape

            dist_estim = nb_utils.NegativeBinomial(mean=tf.exp(a_intercept),
                                                   r=tf.exp(b_intercept),
                                                   name="dist_estim")

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
            log_count_probs = dist_obs.log_prob(sample_data, name="log_count_probs")

            # minimize negative log probability (log(1) = 0);
            # use the mean loss to keep a constant learning rate independently of the batch size
            loss = -tf.reduce_mean(log_count_probs, name="loss")

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
            self.loss = loss


class EstimatorGraph(TFEstimatorGraph):
    sample_data: tf.Tensor

    mu: tf.Tensor
    sigma2: tf.Tensor
    a: tf.Tensor
    b: tf.Tensor

    def __init__(
            self,
            num_samples,
            num_genes,
            num_design_params,
            graph=None,
            batch_size=250
    ):
        super().__init__(graph)

        # initial graph elements
        with self.graph.as_default():
            sample_data = tf_utils.caching_placeholder(tf.float32, shape=(num_samples, num_genes), name="sample_data")
            design = tf_utils.caching_placeholder(tf.float32, shape=(num_samples, num_design_params), name="design")

            learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
            # train_steps = tf.placeholder(tf.int32, shape=(), name="training_steps")

            with tf.name_scope("initialization"):
                # implicit broadcasting of sample_data and initial_mixture_probs to
                #   shape (num_mixtures, num_samples, num_genes)
                init_dist = nb_utils.fit(sample_data, axis=-2)
                assert init_dist.r.shape == [1, num_genes]

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
                batch_sample_data,
                batch_design
            )

            with tf.name_scope("mu_estim"):
                mu_estim = tf.exp(tf.tile(batch_model.a_intercept, (num_samples, 1)))
            with tf.name_scope("r_estim"):
                r_estim = tf.exp(tf.tile(batch_model.b_intercept, (num_samples, 1)))
            dist_estim = nb_utils.NegativeBinomial(mean=mu_estim, r=r_estim)

            with tf.name_scope("mu_obs"):
                mu_obs = tf.exp(tf.matmul(design, batch_model.a))
            with tf.name_scope("r_obs"):
                r_obs = tf.exp(tf.matmul(design, batch_model.b))
            dist_obs = nb_utils.NegativeBinomial(mean=mu_obs, r=r_obs)

            # set up class attributes
            self.sample_data = sample_data
            self.design = design

            self.distribution_estim = dist_estim
            self.distribution_obs = dist_obs
            self.batch_model = batch_model

            self.mu = dist_obs.mean()
            self.sigma2 = dist_obs.variance()
            self.a = batch_model.a
            self.b = batch_model.b
            assert (self.mu.shape == (num_samples, num_genes))
            assert (self.sigma2.shape == (num_samples, num_genes))
            assert (self.a.shape == (num_design_params, num_genes))
            assert (self.b.shape == (num_design_params, num_genes))

            # training
            self.loss = batch_model.loss

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.gradient = self.optimizer.compute_gradients(self.loss)

            self.global_train_step = tf.Variable(0, name="train_step")

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

    def initialize(self, session, feed_dict, **kwargs):
        with self.graph.as_default():
            for op in self.initializer_ops:
                session.run(op, feed_dict=feed_dict)

    def train(self, session, feed_dict, *args, steps=1000, learning_rate=0.05, **kwargs):
        print("learning rate: %s" % learning_rate)
        errors = []
        for i in range(steps):
            # feed_dict = feed_dict.copy()
            feed_dict = dict()
            feed_dict["learning_rate:0"] = learning_rate
            (train_step, loss_res, _) = session.run((self.global_train_step, self.loss, self.train_op),
                                                    feed_dict=feed_dict)
            errors.append(loss_res)
            print("Step: %d\tloss: %f" % (train_step, loss_res))

        return errors


# g = EstimatorGraph(sim.data.design, optimizable_nb=False)
# writer = tf.summary.FileWriter("/tmp/log/...", g.graph)


class Estimator(AbstractEstimator, TFEstimator, metaclass=abc.ABCMeta):
    model: EstimatorGraph

    def __init__(self, input_data: dict, tf_estimator_graph=None, batch_size=250):
        if tf_estimator_graph is None:
            num_design_params = input_data["design"].shape[-1]
            (num_samples, num_genes) = input_data["sample_data"].shape

            tf.reset_default_graph()
            tf_estimator_graph = EstimatorGraph(num_samples, num_genes, num_design_params,
                                                batch_size=batch_size,
                                                graph=tf.get_default_graph())

        TFEstimator.__init__(self, input_data, tf_estimator_graph)

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
