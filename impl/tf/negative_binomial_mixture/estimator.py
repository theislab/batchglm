import abc

import tensorflow as tf

import impl.tf.util as tf_utils
from .external import AbstractEstimator, TFEstimator, TFEstimatorGraph, nb_utils


class EstimatorGraph(TFEstimatorGraph):
    sample_data: tf.Tensor

    r: tf.Tensor
    p: tf.Tensor
    mu: tf.Tensor
    mixture_assignment: tf.Tensor

    def __init__(
            self,
            num_mixtures,
            num_samples,
            num_genes,
            graph=None,
            optimizable_nb=False,
            use_em=False,
            random_effect=0.1,
            learning_rate=0.05
    ):
        super().__init__(graph)

        with self.graph.as_default():
            # placeholders
            sample_data = tf.placeholder(tf.float32, shape=(num_samples, num_genes), name="sample_data")
            initial_mixture_probs = tf.placeholder(tf.float32,
                                                   shape=(num_mixtures, num_samples),
                                                   name="initial_mixture_probs")

            # data preparation
            with tf.name_scope("prepare_data"):
                # apply a random intercept to avoid zero gradients and infinite values
                with tf.name_scope("randomize"):
                    initial_mixture_probs += tf.random_uniform(initial_mixture_probs.shape, 0, random_effect,
                                                               dtype=tf.float32)
                    initial_mixture_probs = initial_mixture_probs / tf.reduce_sum(initial_mixture_probs, axis=0,
                                                                                  keepdims=True)
                    initial_mixture_probs = tf.expand_dims(initial_mixture_probs, -1)
                    initial_mixture_probs = tf.identity(initial_mixture_probs, name="adjusted_initial_mixture_probs")
                assert (initial_mixture_probs.shape == (num_mixtures, num_samples, 1))

                # broadcast sample data to shape (num_mixtures, num_samples, num_genes)
                with tf.name_scope("broadcast"):
                    sample_data = tf.expand_dims(sample_data, axis=0)
                    sample_data = tf.tile(sample_data, (num_mixtures, 1, 1))
                    # TODO: change tf.tile to tf.broadcast_to after next TF release
                assert (sample_data.shape == (num_mixtures, num_samples, num_genes))

            # define mixture_prob tensor depending on optimization method
            mixture_prob = None
            with tf.name_scope("mixture_prob"):
                if use_em:
                    mixture_prob = tf.Variable(initial_mixture_probs,
                                               name="mixture_prob",
                                               trainable=False
                                               )
                else:
                    # optimize logits to keep `mixture_prob` between the interval [0, 1]
                    logit_mixture_prob = tf.Variable(tf_utils.logit(initial_mixture_probs), name="logit_prob")
                    mixture_prob = tf.sigmoid(logit_mixture_prob, name="prob")

                    # normalize: the assignment probabilities should sum up to 1
                    # => `sum(mixture_prob of one sample) = 1`
                    mixture_prob = mixture_prob / tf.reduce_sum(mixture_prob, axis=0, keepdims=True)
                    mixture_prob = tf.identity(mixture_prob, name="normalize")
            assert (mixture_prob.shape == (num_mixtures, num_samples, 1))

            # calculate whole model probability:
            distribution = nb_utils.fit(sample_data=sample_data,
                                        axis=-2,
                                        weights=mixture_prob,
                                        name="fit_nb-dist")

            count_probs = distribution.prob(sample_data, name="count_probs")
            log_count_probs = tf.log(count_probs, name="log_count_probs")
            # sum up: for k in num_mixtures: mixture_prob(k) * P(r_k, mu_k, sample_data)
            joint_probs = tf_utils.reduce_weighted_mean(count_probs, weight=mixture_prob, axis=-3,
                                                        name="joint_probs")

            log_probs = tf.log(joint_probs, name="log_probs")

            # define training operations
            with tf.name_scope("training"):
                # minimize negative log probability (log(1) = 0)
                loss = -tf.reduce_sum(log_probs, name="loss")

                # define train function
                em_op = None
                if use_em:
                    with tf.name_scope("expectation_maximization"):
                        r"""
                        E(p_{j,k}) = \frac{P_{x}(j,k)}{\sum_{a}{P_{x}(j,a)}} \\
                        P_{x}(j,k) = \prod_{i}{L_{NB}(x_{i,j,k} | \mu_{j,k}, \phi_{j,k})} \\
                        log_{P_x}(j, k) = \sum_{i}{log(L_{NB}(x_{i,j,k} | \mu_{j,k}, \phi_{j,k}))} \\
                        E(p_{j,k}) = exp(\frac{log_{P_{x}(j,k)}}{log(\sum_{a}{exp(log_{P_{x}}(j,a)}))})

                        Here, the log(sum(exp(a))) trick can be used for the denominator to avoid numeric instabilities.
                        """
                        sum_of_logs = tf.reduce_sum(log_count_probs, axis=-1, keepdims=True)
                        expected_weight = sum_of_logs - tf.reduce_logsumexp(sum_of_logs, axis=0, keepdims=True)
                        expected_weight = tf.exp(expected_weight)
                        expected_weight = tf.identity(expected_weight, name="normalize")

                        em_op = tf.assign(mixture_prob, expected_weight)
                train_op = None
                if optimizable_nb or not use_em:
                    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
                    if use_em:
                        train_op = tf.group(train_op, em_op)
                else:
                    train_op = em_op

            # parameters
            with tf.name_scope("mu"):
                mu = tf.reduce_sum(distribution.mu * mixture_prob, axis=-3)
            with tf.name_scope("r"):
                r = tf.reduce_sum(distribution.r * mixture_prob, axis=-3)
            with tf.name_scope("p"):
                p = tf.reduce_sum(distribution.p * mixture_prob, axis=-3)
            log_mu = tf.log(mu, name="log_mu")
            log_r = tf.log(r, name="log_r")
            log_p = tf.log(p, name="log_p")

            # set up class attributes
            self.sample_data = sample_data

            self.initializer_op = tf.global_variables_initializer()

            self.r = r
            self.p = p
            self.mu = mu
            self.log_r = log_r
            self.log_p = log_p
            self.log_mu = log_mu
            assert (self.r.shape == (num_samples, num_genes))
            assert (self.p.shape == (num_samples, num_genes))
            assert (self.mu.shape == (num_samples, num_genes))

            self.mixture_prob = tf.squeeze(mixture_prob)
            with tf.name_scope("mixture_assignment"):
                self.mixture_assignment = tf.squeeze(tf.argmax(mixture_prob, axis=0))
            assert (self.mixture_prob.shape == (num_mixtures, num_samples))
            assert (self.mixture_assignment.shape == (num_samples))

            self.distribution = distribution
            self.log_probs = log_probs

            self.loss = loss
            self.train_op = train_op

    def initialize(self, session, feed_dict, **kwargs):
        session.run(self.initializer_op, feed_dict=feed_dict)

    def train(self, session, feed_dict, *args, steps=1000, **kwargs):
        errors = []
        for i in range(steps):
            (loss_res, _) = session.run((self.loss, self.train_op),
                                        feed_dict=feed_dict)
            errors.append(loss_res)
            print(i)

        return errors


# g = EstimatorGraph(2, 2000, 10000, optimizable_nb=False)
# writer = tf.summary.FileWriter("/tmp/log/...", g.graph)
# writer.close()


class Estimator(AbstractEstimator, TFEstimator, metaclass=abc.ABCMeta):
    model: EstimatorGraph

    def __init__(self, input_data: dict, use_em=False, tf_estimator_graph=None):
        if tf_estimator_graph is None:
            num_mixtures = input_data["initial_mixture_probs"].shape[0]
            (num_samples, num_genes) = input_data["sample_data"].shape
            tf_estimator_graph = EstimatorGraph(num_mixtures, num_samples, num_genes, use_em=use_em)

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
    def mixture_assignment(self):
        return self.run(self.model.mixture_assignment)

    @property
    def mixture_prob(self):
        return self.run(self.model.mixture_prob)
