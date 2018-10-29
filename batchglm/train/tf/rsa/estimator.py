import abc
from typing import Union, Dict, Tuple, List
import logging
import pprint
from enum import Enum

import tensorflow as tf
# import tensorflow_probability as tfp

import numpy as np

try:
    import anndata
except ImportError:
    anndata = None

from .external import AbstractEstimator, XArrayEstimatorStore, InputData, Model, MonitoredTFEstimator, TFEstimatorGraph
from .external import nb_utils, train_utils, op_utils, rand_utils
from .external import pkg_constants




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
            sample_data,
            design,
            num_mixtures,
            num_samples,
            num_genes,
            num_design_params,
            batch_size=250,
            graph=None,
            random_effect=0.1,
    ):
        super().__init__(graph)

        # initial graph elements
        with self.graph.as_default():
            # ### placeholders
            # sample_data = tf_utils.caching_placeholder(tf.float32, shape=(num_samples, num_genes), name="sample_data")
            # design = tf_utils.caching_placeholder(tf.float32, shape=(num_samples, num_design_params), name="design")
            # initial_mixture_probs = tf_utils.caching_placeholder(tf.float32,
            #                                                      shape=(num_mixtures, num_samples),
            #                                                      name="initial_mixture_probs")

            sample_data = tf.convert_to_tensor(sample_data, dtype=tf.float32, name="sample_data")
            design = tf.convert_to_tensor(design, dtype=tf.float32, name="design")
            assert sample_data.shape == (num_samples, num_genes)
            assert design.shape == (num_samples, num_design_params)

            learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
            # train_steps = tf.placeholder(tf.int32, shape=(), name="training_steps")

            # ### core model
            # # apply a random intercept to avoid zero gradients and infinite values
            # with tf.name_scope("randomize"):
            #     initial_mixture_probs += tf.random_uniform(initial_mixture_probs.shape, 0, random_effect,
            #                                                dtype=tf.float32)
            #     initial_mixture_probs = initial_mixture_probs / tf.reduce_sum(initial_mixture_probs, axis=0,
            #                                                                   keepdims=True)

            with tf.name_scope("broadcast"):
                design_bcast = tf.expand_dims(design, axis=0)
                design_bcast = tf.tile(design_bcast, (num_mixtures, 1, 1))
                assert (design_bcast.shape == (num_mixtures, num_samples, num_design_params))

            with tf.name_scope("initialization"):
                initial_mixture_probs = tf.random_uniform((num_mixtures, num_samples), 0, 1, dtype=tf.float32)
                initial_mixture_probs = initial_mixture_probs / tf.reduce_sum(initial_mixture_probs, axis=0,
                                                                              keepdims=True)

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
            data = data.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=batch_size * 5))
            # data = data.repeat()
            # data = data.shuffle(batch_size * 5)
            data = data.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

            iterator = data.make_one_shot_iterator()
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

            # ### management
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

            with tf.name_scope("training"):
                self.global_train_step = tf.train.get_or_create_global_step()
                self.loss = batch_model.loss

                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                self.gradient = self.optimizer.compute_gradients(self.loss)

                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_train_step)

            # self.initializer_ops = [
            #     tf.variables_initializer([sample_data, design]),
            #     iterator.initializer,
            #     tf.variables_initializer(tf.global_variables()),
            # ]
            # with tf.control_dependencies([iterator.initializer]):
            #     self.init_op = tf.variables_initializer(tf.global_variables(), name="init_op")

            self.init_ops = []
            self.init_op = tf.variables_initializer(tf.global_variables(), name="init_op")

            self.saver = tf.train.Saver()
            self.merged_summary = tf.summary.merge_all()
            # self.summary_writer = tf.summary.FileWriter(log_dir, self.graph)

            # ### set up class attributes
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

    def initialize(self, session, feed_dict, **kwargs):
        with self.graph.as_default():
            for op in self.init_ops:
                session.run(op, feed_dict=feed_dict)
            session.run(self.init_op)

    def train(self, session, feed_dict=None, *args, steps=1000, learning_rate=0.05, **kwargs):
        print("learning rate: %s" % learning_rate)

        feed_dict = dict() if feed_dict is None else feed_dict.copy()
        feed_dict["learning_rate:0"] = learning_rate

        loss_res = None
        for i in range(steps):
            (train_step, loss_res, _) = session.run((self.global_train_step, self.loss, self.train_op),
                                                    feed_dict=feed_dict)

            # if train_step % 10 == 0:  # Record summaries and test-set accuracy
            #     summary, = session.run([self.merged_summary])
            #     self.summary_writer.add_summary(summary, train_step)
            #     self.save(session)

            print("Step: %d\tloss: %f" % (train_step, loss_res))

        return loss_res

    # def save(self, session):
    #     self.saver.save(session, os.path.join(self.log_dir, "state"), global_step=self.global_train_step)
    #
    # def restore(self, session, state_file):
    #     self.saver.restore(session, state_file)


# g = EstimatorGraph(sim.data.design, optimizable_nb=False)
# writer = tf.summary.FileWriter("/tmp/log/...", g.graph)


class Estimator(AbstractEstimator, TFEstimator, metaclass=abc.ABCMeta):
    model: EstimatorGraph

    scaffold: tf.train.Scaffold
    hooks: List[tf.train.SessionRunHook]

    def __init__(self, input_data: xr.Dataset,
                 batch_size=500,
                 num_mixtures=2,
                 initial_mixtures=None,
                 fixed_mixtures=None,
                 model=None,
                 working_dir=None):
        if working_dir is None:
            working_dir = os.path.join("data/log/", self.__module__,
                                       datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
            if not os.path.exists(working_dir):
                os.makedirs(working_dir)
        self.working_dir = working_dir

        if model is None:
            # num_mixtures = input_data.dims["mixtures"]
            num_genes = input_data.dims["genes"]
            num_samples = input_data.dims["samples"]
            num_design_params = input_data.dims["design_params"]

            tf.reset_default_graph()

            model = EstimatorGraph(np.asarray(input_data["sample_data"], dtype=np.float32),
                                   np.asarray(input_data["design"], dtype=np.float32),
                                   num_mixtures, num_samples, num_genes, num_design_params,
                                   batch_size=batch_size,
                                   graph=tf.get_default_graph())

        with model.graph.as_default():
            self.scaffold = tf.train.Scaffold(
                init_op=model.init_op,
                summary_op=model.merged_summary,
                saver=model.saver,
            )
            self.hooks = [
                TimedRunHook(
                    run_steps=11,
                    call_request_tensors={p: model.__getattribute__(p) for p in PARAMS.keys()},
                    call_fn=lambda sess, step, data: self._save_timestep(step, data)
                ),
                tf.train.NanTensorHook(model.loss),
            ]

        TFEstimator.__init__(self, input_data, model)

    def create_new_session(self):
        self.close_session()
        self.feed_dict = {}
        with self.model.graph.as_default():
            self.session = tf.train.MonitoredTrainingSession(
                checkpoint_dir=self.working_dir,
                scaffold=self.scaffold,
                hooks=self.hooks,
                save_checkpoint_steps=10,
                save_summaries_steps=10,
            )

    def _save_timestep(self, step, data: dict):
        xarray = {key: (dim, data[key]) for (key, dim) in PARAMS.items()}
        xarray = xr.Dataset(xarray)
        xarray["global_step"] = (), step
        xarray["time"] = (), datetime.datetime.now()

        xarray.to_netcdf(path=os.path.join(self.working_dir, "estimation-%d.h5" % step))

    def to_xarray(self):
        model_params = {p: self.model.__getattribute__(p) for p in PARAMS.keys()}

        data = self.session.run(model_params, feed_dict=None)

        output = {key: (dim, data[key]) for (key, dim) in PARAMS.items()}
        output = xr.Dataset(output)

        return output

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
