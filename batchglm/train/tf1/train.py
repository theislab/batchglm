import contextlib
import logging
import tensorflow as tf
from typing import Union, Dict

logger = logging.getLogger(__name__)


class MultiTrainer:

    def __init__(
            self,
            learning_rate,
            loss=None,
            variables: tf.Variable = None,
            gradients: tf.Tensor = None,
            apply_gradients: Union[callable, Dict[tf.Variable, callable]] = None,
            newton_delta: tf.Tensor = None,
            irls_delta: tf.Tensor = None,
            irls_gd_delta: tf.Tensor = None,
            train_ops_nr_tr=None,
            train_ops_irls_tr=None,
            train_ops_irls_gd_tr=None,
            global_step=None,
            apply_train_ops: callable = None,
            provide_optimizers: Union[dict, None] = None,
            session = None,
            name=None
    ):
        r"""

        :param learning_rate: learning rate used for training
        :param loss: loss which should be minimized
        :param variables: list of variables which will be trained
        :param gradients: tensor of gradients of loss function with respect to trained parameters.
            If gradients is not given, gradients are computed via tensorflow based on the given loss.
        :param apply_gradients: callable(s) appliable to the gradients.
            Can be either a single callable which will be applied to all gradients or a dict of
            {tf1.Variable: callable} mappings.
        :param newton_delta: tensor Precomputed custom newton-rhapson parameter update to apply.
        :param irls_delta: tensor Precomputed custom IRLS parameter update to apply.
        :param global_step: global step counter
        :param apply_train_ops: callable which will be applied to all train ops
        :param name: optional name scope
        """
        self.session = session
        with contextlib.ExitStack() as stack:
            if name is not None:
                gs = stack.enter_context(tf.name_scope(name))

            if gradients is None:
                if variables is None:
                    raise ValueError("Either variables and loss or gradients have to be specified")

                logger.debug(" **** Compute gradients using tensorflow")
                plain_gradients = tf.gradients(loss, variables)
                plain_gradients_vars = [(g, v) for g, v in zip(plain_gradients, variables)]
            else:
                plain_gradients_vars = [(gradients, variables)]

            if callable(apply_gradients):
                gradients_vars = [(apply_gradients(g), v) for g, v in plain_gradients_vars]
            elif isinstance(apply_gradients, dict):
                gradients_vars = [(apply_gradients[v](g) if v in apply_gradients else g, v) for g, v in plain_gradients_vars]
            else:
                gradients_vars = plain_gradients_vars

            # Standard tensorflow optimizers.
            if provide_optimizers["gd"]:
                logger.debug(" *** Building optimizer: GD")
                optim_GD = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
                train_op_GD = optim_GD.apply_gradients(gradients_vars, global_step=global_step)
                if apply_train_ops is not None:
                    train_op_GD = apply_train_ops(train_op_GD)
                update_op_GD = tf.multiply(gradients, learning_rate)
            else:
                optim_GD = None
                train_op_GD = None
                update_op_GD = None

            if provide_optimizers["adam"]:
                logger.debug(" *** Building optimizer: ADAM")
                optim_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
                train_op_Adam = optim_Adam.apply_gradients(gradients_vars, global_step=global_step)
                if apply_train_ops is not None:
                    train_op_Adam = apply_train_ops(train_op_Adam)
                update_op_Adam = tf.multiply(gradients, learning_rate)  # TODO replace by actual step
            else:
                optim_Adam = None
                train_op_Adam = None
                update_op_Adam = None

            if provide_optimizers["adagrad"]:
                logger.debug(" *** Building optimizer: ADAGRAD")
                optim_Adagrad = tf.compat.v1.train.AdagradOptimizer(learning_rate=learning_rate)
                train_op_Adagrad = optim_Adagrad.apply_gradients(gradients_vars, global_step=global_step)
                if apply_train_ops is not None:
                    train_op_Adagrad = apply_train_ops(train_op_Adagrad)
                update_op_Adagrad = tf.multiply(gradients, learning_rate)  # TODO replace by actual step
            else:
                optim_Adagrad = None
                train_op_Adagrad = None
                update_op_Adagrad = None

            if provide_optimizers["rmsprop"]:
                logger.debug(" *** Building optimizer: RMSPROP")
                optim_RMSProp = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate)
                train_op_RMSProp = optim_RMSProp.apply_gradients(gradients_vars, global_step=global_step)
                if apply_train_ops is not None:
                    train_op_RMSProp = apply_train_ops(train_op_RMSProp)
                update_op_RMSProp = tf.multiply(gradients, learning_rate)  # TODO replace by actual step
            else:
                optim_RMSProp = None
                train_op_RMSProp = None
                update_op_RMSProp = None

            # Custom optimizers.
            if provide_optimizers["nr"] and newton_delta is not None:
                logger.debug(" *** Building optimizer: NR")
                update_op_nr = newton_delta

                theta_new_nr = variables - newton_delta
                train_op_nr = tf.group(
                    tf.compat.v1.assign(variables, theta_new_nr),
                    tf.compat.v1.assign_add(global_step, 1)
                )
                if apply_train_ops is not None:
                    train_op_nr = apply_train_ops(train_op_nr)
            else:
                train_op_nr = None
                update_op_nr = None

            if provide_optimizers["irls"] and irls_delta is not None:
                logger.debug(" *** Building optimizer: IRLS")
                update_op_irls = irls_delta

                theta_new_irls = variables - irls_delta
                train_op_irls = tf.group(
                    tf.compat.v1.assign(variables, theta_new_irls),
                    tf.compat.v1.assign_add(global_step, 1)
                )
                if apply_train_ops is not None:
                    train_op_irls = apply_train_ops(train_op_irls)
            else:
                train_op_irls = None
                update_op_irls = None

            if provide_optimizers["irls_gd"] and irls_gd_delta is not None:
                logger.debug(" *** Building optimizer: IRLS_GD")
                update_op_irls_gd = irls_gd_delta

                theta_new_irls_gd = variables - irls_gd_delta
                train_op_irls_gd = tf.group(
                    tf.compat.v1.assign(variables, theta_new_irls_gd),
                    tf.compat.v1.assign_add(global_step, 1)
                )
                if apply_train_ops is not None:
                    train_op_irls_gd = apply_train_ops(train_op_irls_gd)
            else:
                train_op_irls_gd = None
                update_op_irls_gd = None

            if provide_optimizers["nr_tr"] and train_ops_nr_tr is not None:
                logger.debug(" *** Building optimizer: NR_TR")
                train_op_nr_tr = {"trial_op": train_ops_nr_tr["trial_op"],
                                  "update_op": tf.group(train_ops_nr_tr["update_op"],
                                                        tf.compat.v1.assign_add(global_step, 1))}
                update_op_nr_tr = train_ops_nr_tr["update"]
            else:
                train_op_nr_tr = None
                update_op_nr_tr = None

            if provide_optimizers["irls_tr"] and train_ops_irls_tr is not None:
                logger.debug(" *** Building optimizer: IRLS_TR")
                train_op_irls_tr = {"trial_op": train_ops_irls_tr["trial_op"],
                                    "update_op": tf.group(train_ops_irls_tr["update_op"],
                                                          tf.compat.v1.assign_add(global_step, 1))}
                update_op_irls_tr = train_ops_irls_tr["update"]
            else:
                train_op_irls_tr = None
                update_op_irls_tr = None

            if provide_optimizers["irls_gd_tr"] and train_ops_irls_gd_tr is not None:
                logger.debug(" *** Building optimizer: IRLS_GD_TR")
                train_op_irls_gd_tr = {"trial_op": train_ops_irls_gd_tr["trial_op"],
                                    "update_op": tf.group(train_ops_irls_gd_tr["update_op"],
                                                          tf.compat.v1.assign_add(global_step, 1))}
                update_op_irls_gd_tr = train_ops_irls_gd_tr["update"]
            else:
                train_op_irls_gd_tr = None
                update_op_irls_gd_tr = None

            self.global_step = global_step
            self.plain_gradients = plain_gradients_vars
            self.gradients = gradients_vars

            self.optim_GD = optim_GD
            self.optim_Adam = optim_Adam
            self.optim_Adagrad = optim_Adagrad
            self.optim_RMSProp = optim_RMSProp

            self.train_op_GD = train_op_GD
            self.train_op_Adam = train_op_Adam
            self.train_op_Adagrad = train_op_Adagrad
            self.train_op_RMSProp = train_op_RMSProp
            self.train_op_nr = train_op_nr
            self.train_op_nr_tr = train_op_nr_tr
            self.train_op_irls = train_op_irls
            self.train_op_irls_gd = train_op_irls_gd
            self.train_op_irls_tr = train_op_irls_tr
            self.train_op_irls_gd_tr = train_op_irls_gd_tr

            self.update_op_GD = update_op_GD
            self.update_op_Adam = update_op_Adam
            self.update_op_Adagrad = update_op_Adagrad
            self.update_op_RMSProp = update_op_RMSProp
            self.update_op_nr = update_op_nr
            self.update_op_nr_tr = update_op_nr_tr
            self.update_op_irls = update_op_irls
            self.update_op_irls_gd = update_op_irls_gd
            self.update_op_irls_tr = update_op_irls_tr
            self.update_op_irls_gd_tr = update_op_irls_gd_tr

            #self.train_op_bfgs = train_op_bfgs


    def train_op_by_name(self, name: str):
        """
        Returns the train op specified by the provided name
        
        :param name: name of the requested train op. Can be:
        
            - "Adam"
            - "Adagrad"
            - "RMSprop"
            - "GradientDescent" or "GD"
        :return: train op
        """
        name_lower = name.lower()
        if name_lower == "gradient_descent" or name_lower == "gd":
            if self.train_op_GD is None:
                raise ValueError("Gradient decent not provided in initialization.")
            return {"train": self.train_op_GD, "update": self.update_op_GD}
        elif name_lower == "adam":
            if self.train_op_Adam is None:
                raise ValueError("Adam not provided in initialization.")
            return {"train": self.train_op_Adam, "update": self.update_op_Adam}
        elif name_lower == "adagrad":
            if self.train_op_Adagrad is None:
                raise ValueError("Adagrad decent not provided in initialization.")
            return {"train": self.train_op_Adagrad, "update": self.update_op_Adagrad}
        elif name_lower == "rmsprop":
            if self.train_op_RMSProp is None:
                raise ValueError("RMSProp decent not provided in initialization.")
            return {"train": self.train_op_RMSProp, "update": self.update_op_RMSProp}
        elif name_lower == "bfgs":
            if self.train_op_bfgs is None:
                raise ValueError("BFGS not provided in initialization.")
            return {"train": self.train_op_bfgs, "update": self.update_op_bfgs}
        elif name_lower.lower() == "newton" or \
                name_lower.lower() == "newton_raphson" or \
                name_lower.lower() == "nr":
            if self.train_op_nr is None:
                raise ValueError("Newton-rhapson not provided in initialization.")
            return {"train": self.train_op_nr, "update": self.update_op_nr}
        elif name_lower.lower() == "newton_tr" or \
                name_lower.lower() == "newton_raphson_tr" or \
                name_lower.lower() == "nr_tr":
            if self.train_op_nr_tr is None:
                raise ValueError("Newton-rhapson trust-region not provided in initialization.")
            return {"train": self.train_op_nr_tr, "update": self.update_op_nr_tr}
        elif name_lower.lower() == "irls" or \
                name_lower.lower() == "iwls":
            if self.train_op_irls is None:
                raise ValueError("IRLS not provided in initialization.")
            return {"train": self.train_op_irls, "update": self.update_op_irls}
        elif name_lower.lower() == "irls_gd" or \
                name_lower.lower() == "iwls_gd":
            if self.train_op_irls_gd is None:
                raise ValueError("IRLS_GD not provided in initialization.")
            return {"train": self.train_op_irls_gd, "update": self.update_op_irls_gd}
        elif name_lower.lower() == "irls_tr" or \
                name_lower.lower() == "iwls_tr":
            if self.train_op_irls_tr is None:
                raise ValueError("IRLS trust-region not provided in initialization.")
            return {"train": self.train_op_irls_tr, "update": self.update_op_irls_tr}
        elif name_lower.lower() == "irls_gd_tr" or \
             name_lower.lower() == "iwls_gd_tr":
            if self.train_op_irls_gd_tr is None:
                raise ValueError("IRLS_GD trust-region not provided in initialization.")
            return {"train": self.train_op_irls_gd_tr, "update": self.update_op_irls_gd_tr}
        else:
                raise ValueError("Unknown optimizer %s" % name)

    def gradient_by_variable(self, variable: tf.Variable):
        """
        Returns the gradient to a specific variable if existing in self.gradients
        :param variable: the variable whose gradient is requested
        :return: gradient tensor or None if not found
        """
        for g, v in self.gradients:
            if v is variable:
                return g
        return None

    def plain_gradient_by_variable(self, variable: tf.Variable):
        """
        Returns the plain gradient to a specific variable if existing in self.plain_gradients
        :param variable: the variable whose gradient is requested
        :return: gradient tensor or None if not found
        """
        for g, v in self.plain_gradients:
            if v is variable:
                return g
        return None
