from importlib import import_module
import logging
import tensorflow as tf
from .external import ModelBase
from .processModel import ProcessModelGLM
logger = logging.getLogger("batchglm")


class GLM(ModelBase, ProcessModelGLM):
    """
    base GLM class containg the model call.
    """

    def __init__(
            self,
            model_vars,
            optimizer: str,
            noise_module: str,
            use_gradient_tape: bool = False,
            compute_a: bool = True,
            compute_b: bool = True,
            dtype: str = "float32",
    ):
        super(GLM, self).__init__(dtype=dtype)

        self.model_vars = model_vars
        self.use_gradient_tape = use_gradient_tape
        self.compute_a = compute_a
        self.compute_b = compute_b
        self.params = tf.Variable(tf.concat(
            [
                model_vars.init_a_clipped,
                model_vars.init_b_clipped,
            ],
            axis=0
        ), name="params", trainable=True)
        self.params_copy = self.params

        # import and add noise model specific layers.
        layers = import_module('...' + noise_module + '.layers', __name__)
        grad_layers = import_module('...' + noise_module + '.layers_gradients', __name__)
        self.unpack_params = layers.UnpackParams(dtype=dtype)
        self.linear_loc = layers.LinearLoc(dtype=dtype)
        self.linear_scale = layers.LinearScale(dtype=dtype)
        self.linker_loc = layers.LinkerLoc(dtype=dtype)
        self.linker_scale = layers.LinkerScale(dtype=dtype)
        self.likelihood = layers.Likelihood(dtype=dtype)
        self.jacobian = grad_layers.Jacobian(model_vars=model_vars, dtype=dtype)
        self.hessian = grad_layers.Hessian(model_vars=model_vars, dtype=dtype)
        self.fim = grad_layers.FIM(model_vars=model_vars, dtype=dtype)

        self.setMethod(optimizer)

    def setMethod(self, optimizer: str):
        """
        Determines which function is executed to calculate and return the desired outputs when
        calling the model. The internal function is chosen based on the given optimizer. It will
        through an AssertionError if the optimizer is not understood.
        """
        optimizer = optimizer.lower()
        if optimizer in ['gd', 'adam', 'adagrad', 'rmsprop']:
            self._calc = self.calc_jacobians

        elif optimizer in ['nr', 'nr_tr']:
            self._calc = self._calc_hessians

        elif optimizer in ['irls', 'irls_tr', 'irls_gd', 'irls_gd_tr', 'irls_ar', 'irls_tr_ar', 'irls_tr_gd_tr']:
            self._calc = self._calc_fim
        else:
            assert False, ("Unrecognized optimizer: %s", optimizer)

    def featurewise_batch(self):
        """
        Applies a boolean mask over the feature dimension of the parameter matrix by removing
        some feature columns (e.g. to exclude converged parameters) determined by the
        `remaining_features` vector in `model_vars`. This method must be called after each
        featurewise batch event to ensure the feature dimension of the input tensors matches the
        feature dimension of `params_copy` in the following model call.
        """
        self.params_copy = tf.Variable(
            tf.boolean_mask(tensor=self.params, mask=self.model_vars.remaining_features, axis=1),
            trainable=True)

    def apply_featurewise_updates(self, full_params_copy: tf.Tensor):
        """
        Applies featurewise updates stored in `params_copy` on `params`. Feature columns in
        `params` corresponding to remaining feature columns in `params_copy` are overwritten with
        the new values while the others (corresponding to features with converged parameters) are
        retained. This method must be called after each featurewise batch event to ensure that the
        updates stored in `params_copy` aren't lost when deriving a new `params_copy` from `params`
        in the following model calls using `featurewise_batch()`.
        """
        self.params.assign(
            tf.where(self.model_vars.remaining_features, full_params_copy, self.params))

    def _call_parameters(self, inputs):
        design_loc, design_scale, size_factors = inputs
        a_var, b_var = self.unpack_params([self.params_copy, self.model_vars.a_var.get_shape()[0]])
        eta_loc = self.linear_loc([a_var, design_loc, self.model_vars.constraints_loc, size_factors])
        eta_scale = self.linear_scale([b_var, design_scale, self.model_vars.constraints_scale])
        loc = self.linker_loc(eta_loc)
        scale = self.linker_scale(eta_scale)
        return eta_loc, eta_scale, loc, scale, a_var, b_var

    def calc_ll(self, inputs):
        """
        Calculates the log probabilities, summed up per feature column and returns it together with
        loc, scale, a_var, and b_var (forwarding results from `_call_parameters`).
        """
        parameters = self._call_parameters(inputs[1:])
        log_probs = self.likelihood([*parameters[:-2], inputs[0]])
        log_probs = tf.reduce_sum(log_probs, axis=0)
        return (log_probs, *parameters[2:])

    def calc_jacobians(self, inputs, compute_a=True, compute_b=None, concat=True):
        if compute_b is None:
            compute_b = self.compute_b
        return self._calc_jacobians(inputs, compute_a=compute_a, compute_b=compute_b, concat=concat)[2:]

    def _calc_jacobians(self, inputs, compute_a, compute_b, concat=True, transpose=True):
        """
        calculates jacobian.

        :param inputs: TODO
        :param concat: boolean
            if true, concatenates the loc and scale block.
        :param transpose: bool
            transpose the gradient if true.
            autograd returns gradients with respect to the shape of self.params.
            But analytic differentiation returns it the other way, which is
            often needed for downstream operations (e.g. hessian)
            Therefore, if self.use_gradient_tape, it will transpose if transpose == False
        """

        with tf.GradientTape(persistent=True) as g:
            log_probs, loc, scale, a_var, b_var = self.calc_ll(inputs)

        if self.use_gradient_tape:

            if compute_a:
                if compute_b:
                    if concat:
                        jacobians = g.gradient(log_probs, self.params_copy)
                        if not transpose:
                            jacobians = tf.transpose(jacobians)
                    else:
                        jac_a = g.gradient(log_probs, a_var)
                        jac_b = g.gradient(log_probs, b_var)
                        if not transpose:
                            jac_a = tf.transpose(jac_a)
                            jac_b = tf.transpose(jac_b)
                else:
                    jac_a = g.gradient(log_probs, a_var)
                    jac_b = tf.zeros((jac_a.get_shape()[0], b_var.get_shape()[1]), b_var.dtype)
                    if concat:
                        jacobians = tf.concat([jac_a, jac_b], axis=0)
                        if not transpose:
                            jacobians = tf.transpose(jacobians)
            else:
                jac_b = g.gradient(log_probs, b_var)
                jac_a = tf.zeros((jac_b.get_shape()[0], a_var.get_shape()[0]), a_var.dtype)
                if concat:
                    jacobians = tf.concat([jac_a, jac_b], axis=0)
                    if not transpose:
                        jacobians = tf.transpose(jacobians)

        else:

            if concat:
                jacobians = self.jacobian([*inputs[0:3], loc, scale, True, compute_a, compute_b])
                if transpose:
                    jacobians = tf.transpose(jacobians)
            else:
                jac_a, jac_b = self.jacobian([*inputs[0:3], loc, scale, False, compute_a, compute_b])

        del g
        if concat:
            return loc, scale, log_probs, tf.negative(jacobians)
        return loc, scale, log_probs, tf.negative(jac_a), tf.negative(jac_b)

    def _calc_hessians(self, inputs, compute_a, compute_b):
        # with tf.GradientTape(persistent=True) as g2:
        loc, scale, log_probs, jacobians = self._calc_jacobians(inputs, compute_a=compute_a, compute_b=compute_b, transpose=False)
        '''
        autograd not yet working. TODO: Search error in the following code:

        if self.use_gradient_tape:

            i = tf.constant(0, tf.int32)
            h_tensor_array = tf.TensorArray(  # hessian slices [:,:,j]
                dtype=self.params_copy.dtype,
                size=self.params_copy.get_shape()[0],
                clear_after_read=False
            )
            while i < self.params_copy.get_shape()[0]:
                grad = g2.gradient(results_arr[i], self.params_copy)
                h_tensor_array.write(index=i, value=grad)
                i += 1

            # h_tensor_array is a TensorArray, reshape this into a tensor so that it can be used
            # in down-stream computation graphs.

            hessians = tf.transpose(tf.reshape(
                h_tensor_array.stack(),
                tf.stack((self.params_copy.get_shape()[0],
                          self.params_copy.get_shape()[0],
                          self.params_copy.get_shape()[1]))
            ), perm=[2, 1, 0])
            hessians = tf.negative(hessians)
        '''
        hessians = tf.negative(self.hessian([*inputs[0:3], loc, scale, True, compute_a, compute_b]))
        return log_probs, jacobians, hessians

    def _calc_fim(self, inputs, compute_a, compute_b):
        loc, scale, log_probs, jac_a, jac_b = self._calc_jacobians(
            inputs,
            compute_a=compute_a,
            compute_b=compute_b,
            concat=False,
            transpose=False)
        fim_a, fim_b = self.fim([*inputs[0:3], loc, scale, False, compute_a, compute_b])
        return log_probs, jac_a, jac_b, fim_a, fim_b

    def call(self, inputs, compute_a=True, compute_b=None):
        """
        Wrapper method to call this model. Depending on the desired calculations specified by the
        `optimizer` arg to `__init__`, it will forward the call to the necessary function to perform
        the right calculations and return all the results.
        """
        if compute_b is None:
            compute_b = self.compute_b
        return self._calc(inputs, compute_a, compute_b)
