import logging

import tensorflow as tf

from .external import ReducableTensorsGLM

logger = logging.getLogger("batchglm")


class ReducableTensorsGLMALL(ReducableTensorsGLM):
    """
    """

    def assemble_tensors(self, idx, data):
        """
        Assemble jacobian of a batch of observations across all features.

        This function runs the data batch (an observation) through the
        model graph and calls the wrappers that compute the
        individual closed forms of the jacobian.

        :param idx: Indices of observations.
        :param data: tuple
            Containing the following parameters:
            - X: tf1.tensor observations x features
                Observation by observation and feature.
            - size_factors: tf1.tensor observations x features
                Model size factors by observation and feature.
            - params: tf1.tensor features x coefficients
                Estimated model variables.
        :return J: tf1.tensor features x coefficients
            Jacobian evaluated on a single observation, provided in data.
        """
        if self.noise_model == "nb":
            from .external_nb import BasicModelGraph
        elif self.noise_model == "norm":
            from .external_norm import BasicModelGraph
        elif self.noise_model == "beta":
            from .external_beta import BasicModelGraph
        else:
            raise ValueError("noise model %s was not recognized" % self.noise_model)

        X, design_loc, design_scale, size_factors = data

        model = BasicModelGraph(
            X=X,
            design_loc=design_loc,
            design_scale=design_scale,
            constraints_loc=self.constraints_loc,
            constraints_scale=self.constraints_scale,
            a_var=self.model_vars.a_var,
            b_var=self.model_vars.b_var,
            dtype=self.model_vars.dtype,
            size_factors=size_factors
        )
        dtype = model.dtype

        if self.compute_jac:
            if self.mode_jac == "analytic":
                jac = self.jac_analytic(model=model)
            elif self.mode_jac == "tf1":
                jac = self.jac_tf(model=model)
            else:
                raise ValueError("mode_jac %s not recognized" % self.mode_jac)
        else:
            jac = tf.zeros((), dtype=dtype)

        if self.compute_hessian:
            if self.mode_hessian == "analytic":
                hessian = self.hessian_analytic(model=model)
            elif self.mode_hessian == "tf1":
                hessian = self.hessian_tf(model=model)
            else:
                raise ValueError("mode_hessian %s not recognized" % self.mode_hessian)
        else:
            hessian = tf.zeros((), dtype=dtype)

        if self.compute_fim_a:
            if self.mode_fim == "analytic":
                fim_a = self.fim_a_analytic(model=model)
            else:
                raise ValueError("mode_fim %s not recognized" % self.mode_fim)
        else:
            fim_a = tf.zeros((), dtype=dtype)

        if self.compute_fim_b:
            if self.mode_fim == "analytic":
                fim_b = self.fim_b_analytic(model=model)
            else:
                raise ValueError("mode_fim %s not recognized" % self.mode_fim)
        else:
            fim_b = tf.zeros((), dtype=dtype)

        if self.compute_ll:
            ll = model.log_likelihood
        else:
            ll = tf.zeros((), dtype=dtype)

        return [jac, hessian, fim_a, fim_b, ll]
