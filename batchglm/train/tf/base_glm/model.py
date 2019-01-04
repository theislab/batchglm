import abc
import logging

import tensorflow as tf
import numpy as np

from .external import AbstractEstimator, ProcessModelBase

logger = logging.getLogger(__name__)

ESTIMATOR_PARAMS = AbstractEstimator.param_shapes().copy()
ESTIMATOR_PARAMS.update({
    "batch_probs": ("batch_observations", "features"),
    "batch_log_probs": ("batch_observations", "features"),
    "batch_log_likelihood": (),
    "full_loss": (),
    "full_gradient": ("features",),
})


class ProcessModelGLM(ProcessModelBase):

    @abc.abstractmethod
    def param_bounds(self, dtype):
        pass


class ModelVarsGLM(ProcessModelGLM):
    """ Build tf.Variables to be optimzed and their constraints.

    a_var and b_var slices of the tf.Variable params which contains
    all parameters to be optimized during model estimation.
    Params is defined across both location and scale model so that 
    the hessian can be computed for the entire model.
    a and b are the clipped parameter values which also contain
    constraints and constrained dependent coefficients which are not
    directly optimized.
    """

    a: tf.Tensor
    b: tf.Tensor
    a_var: tf.Variable
    b_var: tf.Variable
    params: tf.Variable
    converged: np.ndarray

    def __init__(
            self,
            dtype,
            init_a,
            init_b,
            constraints_loc,
            constraints_scale,
            name="ModelVars",
    ):
        """

        :param dtype: Precision used in tensorflow.
        :param init_a: nd.array (mean model size x features)
            Initialisation for all parameters of mean model.
        :param init_b: nd.array (dispersion model size x features)
            Initialisation for all parameters of dispersion model.
        :param constraints_loc: tensor (all parameters x dependent parameters)
            Tensor that encodes how complete parameter set which includes dependent
            parameters arises from indepedent parameters: all = <constraints, indep>.
            This tensor describes this relation for the mean model.
            This form of constraints is used in vector generalized linear models (VGLMs).
        :param constraints_scale: tensor (all parameters x dependent parameters)
            Tensor that encodes how complete parameter set which includes dependent
            parameters arises from indepedent parameters: all = <constraints, indep>.
            This tensor describes this relation for the dispersion model.
            This form of constraints is used in vector generalized linear models (VGLMs).
        :param name: tensorflow subgraph name.
        """
        with tf.name_scope(name):
            with tf.name_scope("initialization"):

                init_a = tf.convert_to_tensor(init_a, dtype=dtype)
                init_b = tf.convert_to_tensor(init_b, dtype=dtype)

                init_a = self.tf_clip_param(init_a, "a")
                init_b = self.tf_clip_param(init_b, "b")

        # Param is the only tf.Variable in the graph.
        # a_var and b_var have to be slices of params.
        params = tf.Variable(tf.concat(
            [
                init_a,
                init_b,
            ],
            axis=0
        ), name="params")

        #params_by_gene = [tf.expand_dims(params[:, i], axis=-1) for i in range(params.shape[1])]
        #a_by_gene = [x[0:init_a.shape[0],:] for x in params_by_gene]
        #b_by_gene = [x[init_a.shape[0]:, :] for x in params_by_gene]
        #a_var = tf.concat(a_by_gene, axis=1)
        #b_var = tf.concat(b_by_gene, axis=1)
        a_var = params[0:init_a.shape[0]]
        b_var = params[init_a.shape[0]:]

        a = tf.matmul(constraints_loc,  a_var)
        b = tf.matmul(constraints_scale,  b_var)

        a_clipped = self.tf_clip_param(a, "a")
        b_clipped = self.tf_clip_param(b, "b")

        self.a = a_clipped
        self.b = b_clipped
        self.a_var = a_var
        self.b_var = b_var
        self.params = params
        # Properties to follow gene-wise convergence.
        self.converged = np.repeat(a=False, repeats=self.params.shape[1])  # Initialise to non-converged.
        self.n_features = self.params.shape[1]
        #self.params_by_gene = params_by_gene
        #self.a_by_gene = a_by_gene
        #self.b_by_gene = b_by_gene

    @abc.abstractmethod
    def param_bounds(self, dtype):
        pass


class BasicModelGraphGLM(ProcessModelGLM):
    """

    """
    X: tf.Tensor
    design_loc: tf.Tensor
    design_scale: tf.Tensor
    constraints_loc: tf.Tensor
    constraints_scale: tf.Tensor

    probs: tf.Tensor
    log_probs: tf.Tensor
    log_likelihood: tf.Tensor
    norm_log_likelihood: tf.Tensor
    norm_neg_log_likelihood: tf.Tensor
    loss: tf.Tensor

    @abc.abstractmethod
    def param_bounds(self, dtype):
        pass
