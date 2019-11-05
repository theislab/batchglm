import logging
from typing import Union

import tensorflow as tf

from batchglm.train.tf1.base_glm.model import ModelVarsGLM

logger = logging.getLogger("batchglm")


class ReducableTensorsGLM:
    """
    """

    noise_model: str
    constraints_loc: tf.Tensor
    constraints_scale: tf.Tensor
    model_vars: ModelVarsGLM
    noise_model: str
    compute_a: bool
    compute_b: bool

    jac: Union[tf.Tensor, None]
    jac_a: Union[tf.Tensor, None]
    jac_b: Union[tf.Tensor, None]
    neg_jac: tf.Tensor
    neg_jac_a: Union[tf.Tensor, None]
    neg_jac_b: Union[tf.Tensor, None]

    hessian: Union[tf.Tensor, None]
    hessian_aa: Union[tf.Tensor, None]
    hessian_bb: Union[tf.Tensor, None]
    neg_hessian: Union[tf.Tensor, None]
    neg_hessian_aa: Union[tf.Tensor, None]
    neg_hessian_bb: Union[tf.Tensor, None]

    fim_a: Union[tf.Tensor, None]
    fim_b: Union[tf.Tensor, None]

    neg_loglikelihood: Union[tf.Tensor, None]

    def __init__(
            self,
            model_vars: ModelVarsGLM,
            noise_model: str,
            constraints_loc,
            constraints_scale,
            sample_indices = None,
            data_set: tf.data.Dataset = None,
            data_batch: tf.Tensor = None,
            mode_jac="analytic",
            mode_hessian="analytic",
            mode_fim="analytic",
            compute_a=True,
            compute_b=True,
            compute_jac=True,
            compute_hessian=True,
            compute_fim=True,
            compute_ll=True
    ):
        """ Return computational graph for jacobian based on mode choice.

        :param batched_data:
            Dataset iterator over mini-batches of data (used for training) or tf1.Tensor of mini-batch.
        :param sample_indices: Indices of samples to be used.
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
        :param mode: str
            Mode by with which hessian is to be evaluated,
            "analytic" uses a closed form solution of the jacobian,
            "tf1" allows for evaluation of the jacobian via the tf1.gradients function.
        :param iterator: bool
            Whether an iterator or a tensor (single yield of an iterator) is given
            in.
        :param jac_a: bool
            Wether to compute Jacobian for a parameters. If both jac_a and jac_b are true,
            the entire jacobian is computed in self.jac.
        :param jac_b: bool
            Wether to compute Jacobian for b parameters. If both jac_a and jac_b are true,
            the entire jacobian is computed in self.jac.
        """
        assert data_set is None or data_batch is None

        self.noise_model = noise_model
        self.model_vars = model_vars
        self.constraints_loc = constraints_loc
        self.constraints_scale = constraints_scale

        self.compute_a = compute_a
        self.compute_b = compute_b

        self.mode_jac = mode_jac
        self.mode_hessian = mode_hessian
        self.mode_fim = mode_fim

        self.compute_jac = compute_jac
        self.compute_hessian = compute_hessian
        self.compute_fim_a = compute_fim and compute_a
        self.compute_fim_b = compute_fim and compute_b
        self.compute_ll = compute_ll

        n_var_all = self.model_vars.params.shape[0]
        n_var_a = self.model_vars.a_var.shape[0]
        n_var_b = self.model_vars.b_var.shape[0]
        dtype = self.model_vars.dtype
        self.dtype = dtype

        def map_fun(idx, data):
            return self.assemble_tensors(
                idx=idx,
                data=data
            )

        def init_fun():
            if self.compute_a and self.compute_b:
                n_var_train = n_var_all
            elif self.compute_a and not self.compute_b:
                n_var_train = n_var_a
            elif not self.compute_a and self.compute_b:
                n_var_train = n_var_b
            else:
                n_var_train = 0

            if self.compute_jac and n_var_train > 0:
                jac_init = tf.zeros([model_vars.n_features, n_var_train], dtype=dtype)
            else:
                jac_init = tf.zeros((), dtype=dtype)

            if self.compute_hessian and n_var_train > 0:
                hessian_init = tf.zeros([model_vars.n_features, n_var_train, n_var_train], dtype=dtype)
            else:
                hessian_init = tf.zeros((), dtype=dtype)

            if self.compute_fim_a:
                fim_a_init = tf.zeros([model_vars.n_features, n_var_a, n_var_a], dtype=dtype)
            else:
                fim_a_init = tf.zeros((), dtype=dtype)
            if self.compute_fim_b:
                fim_b_init = tf.zeros([model_vars.n_features, n_var_b, n_var_b], dtype=dtype)
            else:
                fim_b_init = tf.zeros((), dtype=dtype)

            if self.compute_ll:
                ll_init = tf.zeros([model_vars.n_features], dtype=dtype)
            else:
                ll_init = tf.zeros((), dtype=dtype)

            return jac_init, hessian_init, fim_a_init, fim_b_init, ll_init

        def reduce_fun(old, new):
            return (tf.add(old[0], new[0]),
                    tf.add(old[1], new[1]),
                    tf.add(old[2], new[2]),
                    tf.add(old[3], new[3]),
                    tf.add(old[4], new[4]))

        if data_set is not None:
            set_op = data_set.reduce(
                initial_state=init_fun(),
                reduce_func=lambda old, new: reduce_fun(old, map_fun(new[0], new[1]))
            )
            jac, hessian, fim_a, fim_b, ll = set_op
        elif data_batch is not None:
            set_op = map_fun(
                idx=sample_indices,
                data=data_batch
            )
            jac, hessian, fim_a, fim_b, ll = set_op
        else:
            raise ValueError("supply either data_set or data_batch")

        p_shape_a = self.model_vars.a_var.shape[0]  # This has to be _var to work with constraints.

        # With relay across tf1.Variable:
        # Containers and specific slices and transforms:
        if self.compute_a and self.compute_b:
            if self.compute_jac:
                self.jac = tf.Variable(tf.zeros([self.model_vars.n_features, n_var_all], dtype=dtype), dtype=dtype)
                self.jac_a = self.jac[:, :p_shape_a]
                self.jac_b = self.jac[:, p_shape_a:]
            else:
                self.jac = tf.Variable(tf.zeros((), dtype=dtype), dtype=dtype)
                self.jac_a = self.jac
                self.jac_b = self.jac
            self.jac_train = self.jac

            if self.compute_hessian:
                self.hessian = tf.Variable(tf.zeros([self.model_vars.n_features, n_var_all, n_var_all], dtype=dtype), dtype=dtype)
                self.hessian_aa = self.hessian[:, :p_shape_a, :p_shape_a]
                self.hessian_bb = self.hessian[:, p_shape_a:, p_shape_a:]
            else:
                self.hessian = tf.Variable(tf.zeros((), dtype=dtype), dtype=dtype)
                self.hessian_aa = self.hessian
                self.hessian_bb = self.hessian
            self.hessian_train = self.hessian

            if self.compute_fim_a or self.compute_fim_b:
                self.fim_a = tf.Variable(tf.zeros([self.model_vars.n_features, n_var_a, n_var_a], dtype=dtype), dtype=dtype)
                self.fim_b = tf.Variable(tf.zeros([self.model_vars.n_features, n_var_b, n_var_b], dtype=dtype), dtype=dtype)
            else:
                self.fim_a = tf.Variable(tf.zeros((), dtype=dtype), dtype=dtype)
                self.fim_b = tf.Variable(tf.zeros((), dtype=dtype), dtype=dtype)
        elif self.compute_a and not self.compute_b:
            if self.compute_jac:
                self.jac = tf.Variable(tf.zeros([self.model_vars.n_features, n_var_a], dtype=dtype), dtype=dtype)
                self.jac_a = self.jac
            else:
                self.jac = tf.Variable(tf.zeros((), dtype=dtype), dtype=dtype)
                self.jac_a = self.jac
            self.jac_b = None
            self.jac_train = self.jac_a

            if self.compute_hessian:
                self.hessian = tf.Variable(tf.zeros([model_vars.n_features, n_var_a, n_var_a], dtype=dtype), dtype=dtype)
                self.hessian_aa = self.hessian
            else:
                self.hessian = tf.Variable(tf.zeros((), dtype=dtype), dtype=dtype)
                self.hessian_aa = self.hessian
            self.hessian_bb = None
            self.hessian_train = self.hessian_aa

            if self.compute_fim_a:
                self.fim_a = tf.Variable(tf.zeros([model_vars.n_features, n_var_a, n_var_a], dtype=dtype), dtype=dtype)
            else:
                self.fim_a = tf.Variable(tf.zeros((), dtype=dtype), dtype=dtype)
            self.fim_b = tf.Variable(tf.zeros((), dtype=dtype), dtype=dtype)
        elif not self.compute_a and self.compute_b:
            if self.compute_jac:
                self.jac = tf.Variable(tf.zeros([self.model_vars.n_features, n_var_b], dtype=dtype), dtype=dtype)
                self.jac_b = self.jac
            else:
                self.jac = tf.Variable(tf.zeros((), dtype=dtype), dtype=dtype)
                self.jac_b = self.jac
            self.jac_a = None
            self.jac_train = self.jac_b

            if self.compute_hessian:
                self.hessian = tf.Variable(tf.zeros([model_vars.n_features, n_var_b, n_var_b], dtype=dtype), dtype=dtype)
                self.hessian_bb = self.hessian
            else:
                self.hessian = tf.Variable(tf.zeros((), dtype=dtype), dtype=dtype)
                self.hessian_bb = self.hessian
            self.hessian_aa = None
            self.hessian_train = self.hessian_bb

            self.fim_a = tf.Variable(tf.zeros((), dtype=dtype), dtype=dtype)
            if self.compute_fim_b:
                self.fim_b = tf.Variable(tf.zeros([model_vars.n_features, n_var_b, n_var_b], dtype=dtype), dtype=dtype)
            else:
                self.fim_b = tf.Variable(tf.zeros((), dtype=dtype), dtype=dtype)
        else:
            self.jac = tf.Variable(tf.zeros((), dtype=dtype), dtype=dtype)
            self.jac_a = None
            self.jac_b = None
            self.jac_train = None

            self.hessian = tf.Variable(tf.zeros((), dtype=dtype), dtype=dtype)
            self.hessian_aa = None
            self.hessian_bb = None
            self.hessian_train = None

            self.fim_a = tf.Variable(tf.zeros((), dtype=dtype), dtype=dtype)
            self.fim_b = tf.Variable(tf.zeros((), dtype=dtype), dtype=dtype)

        if self.compute_ll:
            self.ll = tf.Variable(tf.zeros([model_vars.n_features], dtype=dtype), dtype=dtype)
        else:
            self.ll = tf.Variable(tf.zeros((), dtype=dtype), dtype=dtype)

        self.neg_jac = tf.negative(self.jac) if self.jac is not None else None
        self.neg_jac_a = tf.negative(self.jac_a) if self.jac_a is not None else None
        self.neg_jac_b = tf.negative(self.jac_b) if self.jac_b is not None else None
        self.neg_jac_train = tf.negative(self.jac_train) if self.jac_train is not None else None

        self.neg_hessian = tf.negative(self.hessian) if self.hessian is not None else None
        self.neg_hessian_aa = tf.negative(self.hessian_aa) if self.hessian_aa is not None else None
        self.neg_hessian_bb = tf.negative(self.hessian_bb) if self.hessian_bb is not None else None
        self.neg_hessian_train = tf.negative(self.hessian_train) if self.hessian_train is not None else None

        self.neg_ll = tf.negative(self.ll) if self.ll is not None else None

        # Setting operation:
        jac_set = tf.compat.v1.assign(self.jac, jac)
        hessian_set = tf.compat.v1.assign(self.hessian, hessian)
        fim_a_set = tf.compat.v1.assign(self.fim_a, fim_a)
        fim_b_set = tf.compat.v1.assign(self.fim_b, fim_b)
        ll_set = tf.compat.v1.assign(self.ll, ll)

        self.set = tf.group(
            set_op,
            jac_set,
            hessian_set,
            fim_a_set,
            fim_b_set,
            ll_set
        )

    def assemble_tensors(
        self,
        idx,
        data
    ):
        raise NotImplementedError()

    def jac_analytic(
            self,
            model
    ) -> tf.Tensor:
        raise NotImplementedError()

    def jac_tf(
            self,
            model
    ) -> tf.Tensor:
        raise NotImplementedError()

    def hessian_analytic(
            self,
            model
    ) -> tf.Tensor:
        raise NotImplementedError()

    def hessian_tf(
            self,
            model
    ) -> tf.Tensor:
        raise NotImplementedError()

    def fim_analytic(
            self,
            model
    ) -> tf.Tensor:
        raise NotImplementedError()