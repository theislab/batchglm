from typing import Union
import logging

import tensorflow as tf
import numpy as np

from .external import GradientGraphGLM, NewtonGraphGLM, TrainerGraphGLM
from .external import EstimatorGraphGLM, FullDataModelGraphGLM
from .external import op_utils
from .external import pkg_constants

logger = logging.getLogger(__name__)


class FullDataModelGraph(FullDataModelGraphGLM):
    """
    Computational graph to evaluate negative binomial GLM on full data set.
    """

    def __init__(
            self,
            sample_indices: tf.Tensor,
            fetch_fn,
            batch_size: Union[int, tf.Tensor],
            model_vars,
            constraints_loc,
            constraints_scale,
            train_a,
            train_b,
            noise_model: str,
            dtype
    ):
        """
        :param sample_indices:
            TODO
        :param fetch_fn:
            TODO
        :param batch_size: int
            Size of mini-batches used.
        :param model_vars: ModelVars
            Variables of model. Contains tf.Variables which are optimized.
        :param constraints_loc: np.ndarray (constraints on mean model x mean model parameters)
            Constraints for location model.
            Array with constraints in rows and model parameters in columns.
            Each constraint contains non-zero entries for the a of parameters that
            has to sum to zero. This constraint is enforced by binding one parameter
            to the negative sum of the other parameters, effectively representing that
            parameter as a function of the other parameters. This dependent
            parameter is indicated by a -1 in this array, the independent parameters
            of that constraint (which may be dependent at an earlier constraint)
            are indicated by a 1.
        :param constraints_scale: np.ndarray (constraints on mean model x mean model parameters)
            Constraints for scale model.
            Array with constraints in rows and model parameters in columns.
            Each constraint contains non-zero entries for the a of parameters that
            has to sum to zero. This constraint is enforced by binding one parameter
            to the negative sum of the other parameters, effectively representing that
            parameter as a function of the other parameters. This dependent
            parameter is indicated by a -1 in this array, the independent parameters
            of that constraint (which may be dependent at an earlier constraint)
            are indicated by a 1.
        :param train_mu: bool
            Whether to train mean model. If False, the initialisation is kept.
        :param train_r: bool
            Whether to train dispersion model. If False, the initialisation is kept.
        :param dtype: Precision used in tensorflow.
        """
        if noise_model == "nb":
            from .external_nb import BasicModelGraph, Jacobians, Hessians
        else:
            raise ValueError("noise model not rewcognized")
        self.noise_model = noise_model

        dataset = tf.data.Dataset.from_tensor_slices(sample_indices)

        batched_data = dataset.batch(batch_size)
        batched_data = batched_data.map(fetch_fn, num_parallel_calls=pkg_constants.TF_NUM_THREADS)
        batched_data = batched_data.prefetch(1)

        def map_model(idx, data) -> BasicModelGraph:
            X, design_loc, design_scale, size_factors = data
            model = BasicModelGraph(
                X=X,
                design_loc=design_loc,
                design_scale=design_scale,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                a=model_vars.a,
                b=model_vars.b,
                dtype=dtype,
                size_factors=size_factors)
            return model

        model = map_model(*fetch_fn(sample_indices))

        with tf.name_scope("log_likelihood"):
            log_likelihood = op_utils.map_reduce(
                last_elem=tf.gather(sample_indices, tf.size(sample_indices) - 1),
                data=batched_data,
                map_fn=lambda idx, data: map_model(idx, data).log_likelihood,
                parallel_iterations=1,
            )
            norm_log_likelihood = log_likelihood / tf.cast(tf.size(sample_indices), dtype=log_likelihood.dtype)
            norm_neg_log_likelihood = - norm_log_likelihood

        with tf.name_scope("loss"):
            loss = tf.reduce_sum(norm_neg_log_likelihood)

        with tf.name_scope("hessians"):
            # Hessian of full model for reporting.
            hessians_full = Hessians(
                batched_data=batched_data,
                sample_indices=sample_indices,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                model_vars=model_vars,
                mode=pkg_constants.HESSIAN_MODE,
                noise_model=noise_model,
                iterator=True,
                hess_a=True,
                hess_b=True,
                dtype=dtype
            )
            # Hessian of submodel which is to be trained.
            if not train_a or not train_b:
                hessians_train = Hessians(
                    batched_data=batched_data,
                    sample_indices=sample_indices,
                    constraints_loc=constraints_loc,
                    constraints_scale=constraints_scale,
                    model_vars=model_vars,
                    mode=pkg_constants.HESSIAN_MODE,
                    noise_model=noise_model,
                    iterator=True,
                    hess_a=train_a,
                    hess_b=train_b,
                    dtype=dtype
                )
            else:
                hessians_train = hessians_full

        with tf.name_scope("jacobians"):
            # Jacobian of full model for reporting.
            jacobian_full = Jacobians(
                batched_data=batched_data,
                sample_indices=sample_indices,
                batch_model=None,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale,
                model_vars=model_vars,
                mode=pkg_constants.JACOBIAN_MODE,
                iterator=True,
                jac_a=True,
                jac_b=True,
                dtype=dtype
            )
            # Jacobian of submodel which is to be trained.
            if not train_a or not train_b:
                jacobian_train = Jacobians(
                    batched_data=batched_data,
                    sample_indices=sample_indices,
                    batch_model=None,
                    constraints_loc=constraints_loc,
                    constraints_scale=constraints_scale,
                    model_vars=model_vars,
                    mode=pkg_constants.JACOBIAN_MODE,
                    iterator=True,
                    jac_a=train_a,
                    jac_b=train_b,
                    dtype=dtype
                )
            else:
                jacobian_train = jacobian_full

        self.X = model.X
        self.design_loc = model.design_loc
        self.design_scale = model.design_scale

        self.batched_data = batched_data

        self.mu = model.mu
        self.r = model.r
        self.sigma2 = model.sigma2

        self.probs = model.probs
        self.log_probs = model.log_probs

        # custom
        self.sample_indices = sample_indices

        self.log_likelihood = log_likelihood
        self.norm_log_likelihood = norm_log_likelihood
        self.norm_neg_log_likelihood = norm_neg_log_likelihood
        self.loss = loss

        self.jac = jacobian_full.jac
        self.neg_jac = jacobian_full.neg_jac
        self.hessian = hessians_full.hessian
        self.neg_hessian = hessians_full.neg_hessian

        self.neg_jac_train = jacobian_train.neg_jac
        self.neg_hessian_train = hessians_train.neg_hessian


class EstimatorGraphAll(EstimatorGraphGLM):
    """

    """
    mu: tf.Tensor
    sigma2: tf.Tensor

    def __init__(
            self,
            fetch_fn,
            feature_isnonzero,
            num_observations,
            num_features,
            num_design_loc_params,
            num_design_scale_params,
            graph: tf.Graph = None,
            batch_size: int = None,
            init_a=None,
            init_b=None,
            constraints_loc: Union[np.ndarray, None] = None,
            constraints_scale: Union[np.ndarray, None] = None,
            train_loc: bool = True,
            train_scale: bool = True,
            provide_optimizers: dict = {"gd": True, "adam": True, "adagrad": True, "rmsprop": True, "nr": True},
            termination_type: str = "global",
            extended_summary=False,
            noise_model: str = None,
            dtype="float32"
    ):
        """

        :param fetch_fn:
            TODO
        :param feature_isnonzero:
            Whether all observations of a feature are zero. Features for which this
            is the case are not fitted.
        :param num_observations: int
            Number of observations.
        :param num_features: int
            Number of features.
        :param num_design_loc_params: int
            Number of parameters per feature in mean model.
        :param num_design_scale_params: int
            Number of parameters per feature in scale model.
        :param graph: tf.Graph
        :param batch_size: int
            Size of mini-batches used.
        :param init_a: nd.array (mean model size x features)
            Initialisation for all parameters of mean model.
        :param init_b: nd.array (dispersion model size x features)
            Initialisation for all parameters of dispersion model.
        :param constraints_loc: Constraints for location model.
            Array with constraints in rows and model parameters in columns.
            Each constraint contains non-zero entries for the a of parameters that
            has to sum to zero. This constraint is enforced by binding one parameter
            to the negative sum of the other parameters, effectively representing that
            parameter as a function of the other parameters. This dependent
            parameter is indicated by a -1 in this array, the independent parameters
            of that constraint (which may be dependent at an earlier constraint)
            are indicated by a 1.
        :param constraints_scale: Constraints for scale model.
            Array with constraints in rows and model parameters in columns.
            Each constraint contains non-zero entries for the a of parameters that
            has to sum to zero. This constraint is enforced by binding one parameter
            to the negative sum of the other parameters, effectively representing that
            parameter as a function of the other parameters. This dependent
            parameter is indicated by a -1 in this array, the independent parameters
            of that constraint (which may be dependent at an earlier constraint)
            are indicated by a 1.
        :param train_loc: bool
            Whether to train mean model. If False, the initialisation is kept.
        :param train_scale: bool
            Whether to train dispersion model. If False, the initialisation is kept.
        :param provide_optimizers:
        :param termination_type:
        :param extended_summary:
        :param dtype: Precision used in tensorflow.
        """
        if noise_model == "nb":
            from .external_nb import BasicModelGraph, ModelVars, Jacobians, Hessians
        else:
            raise ValueError("noise model not rewcognized")
        self.noise_model = noise_model

        EstimatorGraphGLM.__init__(
            self=self,
            num_observations=num_observations,
            num_features=num_features,
            num_design_loc_params=num_design_loc_params,
            num_design_scale_params=num_design_scale_params,
            graph=graph,
            batch_size=batch_size
        )

        # initial graph elements
        with self.graph.as_default():
            # ### placeholders
            learning_rate = tf.placeholder(dtype, shape=(), name="learning_rate")
            # train_steps = tf.placeholder(tf.int32, shape=(), name="training_steps")

            # ### performance related settings
            buffer_size = 4

            with tf.name_scope("input_pipeline"):
                logger.debug(" ** Build input pipeline")
                data_indices = tf.data.Dataset.from_tensor_slices((
                    tf.range(num_observations, name="sample_index")
                ))
                training_data = data_indices.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=2 * batch_size))
                training_data = training_data.batch(batch_size, drop_remainder=True)
                training_data = training_data.map(tf.contrib.framework.sort)  # sort indices
                training_data = training_data.map(fetch_fn, num_parallel_calls=pkg_constants.TF_NUM_THREADS)
                training_data = training_data.prefetch(buffer_size)

                iterator = training_data.make_one_shot_iterator()

                batch_sample_index, batch_data = iterator.get_next()
                (batch_X, batch_design_loc, batch_design_scale, batch_size_factors) = batch_data

            model_vars = ModelVars(
                dtype=dtype,
                init_a=init_a,
                init_b=init_b,
                constraints_loc=constraints_loc,
                constraints_scale=constraints_scale
            )

            with tf.name_scope("batch"):
                logger.debug(" ** Build batched data model")
                # Batch model:
                #   only `batch_size` observations will be used;
                #   All per-sample variables have to be passed via `data`.
                #   Sample-independent variables (e.g. per-feature distributions) can be created inside the batch model
                batch_model = BasicModelGraph(
                    X=batch_X,
                    design_loc=batch_design_loc,
                    design_scale=batch_design_scale,
                    constraints_loc=constraints_loc,
                    constraints_scale=constraints_scale,
                    a=model_vars.a,
                    b=model_vars.b,
                    dtype=dtype,
                    size_factors=batch_size_factors
                )

                # minimize negative log probability (log(1) = 0);
                # use the mean loss to keep a constant learning rate independently of the batch size
                batch_loss = batch_model.loss

                # Define the jacobian on the batched model for newton-rhapson:
                # (note that these are the Jacobian matrix blocks
                # of the trained subset of parameters).
                batch_jac = Jacobians(
                    batched_data=batch_data,
                    sample_indices=batch_sample_index,
                    batch_model=batch_model,
                    constraints_loc=constraints_loc,
                    constraints_scale=constraints_scale,
                    model_vars=model_vars,
                    mode=pkg_constants.JACOBIAN_MODE,
                    iterator=False,
                    jac_a=train_loc,
                    jac_b=train_scale,
                    dtype=dtype
                )

                # Define the hessian on the batched model for newton-rhapson:
                # (note that these are the Hessian matrix blocks
                # of the trained subset of parameters).
                batch_hessians = Hessians(
                    batched_data=batch_data,
                    sample_indices=batch_sample_index,
                    constraints_loc=constraints_loc,
                    constraints_scale=constraints_scale,
                    model_vars=model_vars,
                    mode=pkg_constants.HESSIAN_MODE,
                    noise_model=noise_model,
                    iterator=False,
                    hess_a=train_loc,
                    hess_b=train_scale,
                    dtype=dtype
                )

            with tf.name_scope("full_data"):
                logger.debug(" ** Build full data model")
                # ### alternative definitions for custom observations:
                sample_selection = tf.placeholder_with_default(tf.range(num_observations),
                                                               shape=(None,),
                                                               name="sample_selection")
                full_data_model = FullDataModelGraph(
                    sample_indices=sample_selection,
                    fetch_fn=fetch_fn,
                    batch_size=batch_size * buffer_size,
                    model_vars=model_vars,
                    constraints_loc=constraints_loc,
                    constraints_scale=constraints_scale,
                    train_a=train_loc,
                    train_b=train_scale,
                    noise_model=noise_model,
                    dtype=dtype
                )
                full_data_loss = full_data_model.loss
                fisher_inv = op_utils.pinv(full_data_model.neg_hessian)

                # with tf.name_scope("hessian_diagonal"):
                #     hessian_diagonal = [
                #         tf.map_fn(
                #             # elems=tf.transpose(hess, perm=[2, 0, 1]),
                #             elems=hess,
                #             fn=tf.diag_part,
                #             parallel_iterations=pkg_constants.TF_LOOP_PARALLEL_ITERATIONS
                #         )
                #         for hess in full_data_model.hessians
                #     ]
                #     fisher_a, fisher_b = hessian_diagonal

                mu = full_data_model.mu
                r = full_data_model.r
                sigma2 = full_data_model.sigma2

                idx_nonconverged = np.where(model_vars.converged == False)[0]

                self.model_vars = model_vars
                self.batch_model = batch_model
                self.learning_rate = learning_rate
                self.loss = batch_loss

                self.batch_jac = batch_jac
                self.batch_hessians = batch_hessians

                self.mu = mu
                self.r = r
                self.sigma2 = sigma2

                self.batch_probs = batch_model.probs
                self.batch_log_probs = batch_model.log_probs
                self.batch_log_likelihood = batch_model.norm_log_likelihood

                self.sample_selection = sample_selection
                self.full_data_model = full_data_model

                self.full_loss = full_data_loss
                self.hessians = full_data_model.hessian
                self.fisher_inv = fisher_inv

                self.idx_nonconverged = idx_nonconverged

                GradientGraphGLM.__init__(
                    self=self,
                    termination_type=termination_type,
                    train_loc=train_loc,
                    train_scale=train_scale
                )
                NewtonGraphGLM.__init__(
                    self=self,
                    termination_type=termination_type,
                    provide_optimizers=provide_optimizers,
                    train_mu=train_loc,
                    train_r=train_scale
                )
                TrainerGraphGLM.__init__(
                    self=self,
                    feature_isnonzero=feature_isnonzero,
                    provide_optimizers=provide_optimizers,
                    dtype=dtype
                )

        with tf.name_scope('summaries'):
            tf.summary.histogram('a', model_vars.a)
            tf.summary.histogram('b', model_vars.b)
            tf.summary.scalar('loss', batch_loss)
            tf.summary.scalar('learning_rate', learning_rate)

            if extended_summary:
                tf.summary.scalar('median_ll',
                                  tf.contrib.distributions.percentile(
                                      tf.reduce_sum(batch_model.log_probs, axis=1),
                                      50.)
                                  )
                summary_full_grad = tf.where(tf.is_nan(full_gradient), tf.zeros_like(full_gradient), full_gradient,
                                             name="full_gradient")
                # TODO: adjust this if gradient is changed
                tf.summary.histogram('batch_gradient', batch_trainers.gradient_by_variable(model_vars.params))
                tf.summary.histogram("full_gradient", summary_full_grad)
                tf.summary.scalar("full_gradient_median", tf.contrib.distributions.percentile(full_gradient, 50.))
                tf.summary.scalar("full_gradient_mean", tf.reduce_mean(full_gradient))

        self.saver = tf.train.Saver()
        self.merged_summary = tf.summary.merge_all()
