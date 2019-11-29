import logging
import tensorflow as tf
import numpy as np
from .external import ModelBase, LossBase
from .processModel import ProcessModelGLM
logger = logging.getLogger("batchglm")


class GLM(ModelBase, ProcessModelGLM):

    """
    base GLM class containg the model call.
    """

    compute_a: bool = True
    compute_b: bool = True

    def __init__(
            self,
            model_vars,
            unpack_params: tf.keras.layers.Layer,
            linear_loc: tf.keras.layers.Layer,
            linear_scale: tf.keras.layers.Layer,
            linker_loc: tf.keras.layers.Layer,
            linker_scale: tf.keras.layers.Layer,
            likelihood: tf.keras.layers.Layer,
            jacobian: tf.keras.layers.Layer,
            hessian: tf.keras.layers.Layer,
            fim: tf.keras.layers.Layer,
            optimizer: str,
            use_gradient_tape: bool = False,
    ):
        super(GLM, self).__init__()
        self.model_vars = model_vars
        self.params = tf.Variable(tf.concat(
            [
                model_vars.init_a_clipped,
                model_vars.init_b_clipped,
            ],
            axis=0
        ), name="params", trainable=True)

        self.unpack_params = unpack_params
        self.linear_loc = linear_loc
        self.linear_scale = linear_scale
        self.linker_loc = linker_loc
        self.linker_scale = linker_scale
        self.likelihood = likelihood
        self.jacobian = jacobian
        self.hessian = hessian
        self.fim = fim
        self.use_gradient_tape = use_gradient_tape
        self.params_copy = None
        self.batch_features = False

        self.calc_jacobian = False
        self.calc_hessian = False
        self.calc_fim = False
        self.concat_grads = True

        self._setParams(optimizer)

    def _setParams(self, optimizer):

        optimizer = optimizer.lower()
        if optimizer in ['gd', 'adam', 'adagrad', 'rmsprop']:
            self.calc_jacobian = True

        elif optimizer in ['nr', 'nr_tr']:
            self.calc_hessian = True

        elif optimizer in ['irls', 'irls_tr', 'irls_gd', 'irls_gd_tr']:
            self.calc_fim = True
            self.concat_grads = False

    def _call_parameters(self, inputs, keep_previous_params_copy=False):
        if not keep_previous_params_copy:
            if self.batch_features:
                self.params_copy = tf.Variable(tf.boolean_mask(tensor=self.params,
                                                               mask=tf.logical_not(self.model_vars.converged),
                                                               axis=1), trainable=True)
            else:
                self.params_copy = self.params
        design_loc, design_scale, size_factors = inputs
        a_var, b_var = self.unpack_params([self.params_copy, self.model_vars.a_var.get_shape()[0]])
        eta_loc = self.linear_loc([a_var, design_loc, self.model_vars.constraints_loc, size_factors])
        eta_scale = self.linear_scale([b_var, design_scale, self.model_vars.constraints_scale])
        loc = self.linker_loc(eta_loc)
        scale = self.linker_scale(eta_scale)
        return eta_loc, eta_scale, loc, scale, a_var, b_var

    def calc_ll(self, inputs, keep_previous_params_copy=False):
        parameters = self._call_parameters(inputs[1:], keep_previous_params_copy)
        log_probs = self.likelihood([*parameters[:-2], inputs[0], np.sum(self.model_vars.updated)])
        return (log_probs, *parameters[2:])

    def _calc_jacobians(self, inputs, concat, transpose=True):
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

            if self.compute_a:
                if self.compute_b:
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
                jacobians = self.jacobian([*inputs[0:3], loc, scale, True])
                if transpose:
                    jacobians = tf.transpose(jacobians)
            else:
                jac_a, jac_b = self.jacobian([*inputs[0:3], loc, scale, False])

        del g
        if concat:
            return loc, scale, log_probs, tf.negative(jacobians)
        return loc, scale, log_probs, tf.negative(jac_a), tf.negative(jac_b)

    def calc_hessians(self, inputs, concat=False):
        # with tf.GradientTape(persistent=True) as g2:
        if concat:
            loc, scale, log_probs, jacobians = self._calc_jacobians(inputs, concat=True, transpose=False)
        else:
            loc, scale, log_probs, jac_a, jac_b = self._calc_jacobians(inputs, concat=False, transpose=False)
        # results_arr = [jacobians[:, i] for i in tf.range(self.params_copy.get_shape()[0])]

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
        # else:
        print('opsdfopdsfpodsfpodsfpo')
        if concat:
            hessians = tf.negative(self.hessian([*inputs[0:3], loc, scale, True]))
            return log_probs, jacobians, hessians

        hes_aa, hes_ab, hes_ba, hes_bb = self.hessian([*inputs[0:3], loc, scale, False])
        return log_probs, jac_a, jac_b, tf.negative(hes_aa), \
            tf.negative(hes_ab), tf.negative(hes_ba), tf.negative(hes_bb)
        # del g2 # need to delete this GradientTape because persistent is True.

    def call(self, inputs, training=False, mask=None):
        # X_data, design_loc, design_scale, size_factors = inputs

        # This is for first order optimizations, which get the full jacobian

        if self.calc_jacobian:
            _, _, log_probs, jacobians = self._calc_jacobians(inputs, concat=self.concat_grads)
            return log_probs, jacobians

        # This is for SecondOrder NR/NR_TR
        if self.calc_hessian:
            results = self.calc_hessians(inputs, concat=self.concat_grads)
            return results
        # This is for SecondOrder IRLS/IRLS_GD/IRLS_TR/IRLS_GD_TR
        if self.calc_fim:
            if self.concat_grads:
                loc, scale, log_probs, jacobians = self._calc_jacobians(inputs, concat=True, transpose=False)
                fims = self.fim([*inputs[0:3], loc, scale, True])

                return log_probs, tf.negative(jacobians), fims
            else:
                loc, scale, log_probs, jac_a, jac_b = self._calc_jacobians(inputs, concat=False, transpose=False)
                fim_a, fim_b = self.fim([*inputs[0:3], loc, scale, False])

                return log_probs, jac_a, jac_b, fim_a, fim_b

        raise ValueError("No gradient calculation specified.")


class LossGLM(LossBase):

    def norm_log_likelihood(self, log_probs):
        return tf.reduce_mean(log_probs, axis=0, name="log_likelihood")

    def norm_neg_log_likelihood(self, log_probs):
        return - self.norm_log_likelihood(log_probs)

    def call(self, y_true, log_probs):
        return tf.reduce_sum(self.norm_neg_log_likelihood(log_probs))
