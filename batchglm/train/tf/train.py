import contextlib
import logging
import sys
import time
import threading
from typing import Union, Dict, Callable, List

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from .external import pkg_constants

logger = logging.getLogger(__name__)

class TimedRunHook(tf.train.SessionRunHook):
    """Runs ops or functions every N steps or seconds."""

    _time_measures: list
    _start_time: float

    _next_step: int
    _global_step_tensor: tf.Tensor

    _threads: List[threading.Thread]

    def __init__(self,
                 run_steps=None,
                 step_offset=1,
                 run_secs=None,
                 call_request_tensors: Dict[str, tf.Tensor] = None,
                 call_fn: Callable = None,
                 asynchronous: bool = False,
                 ):
        """Initializes a `TimedRunHook`.

        :param run_steps: `int`, fire `call_fn` every N steps. Exactly one of
            `run_secs` and `run_steps` should be set.
        :param step_offset: If specified, `call_fn` will be fired every N + `step_offset` step
        :param run_secs: `int`, run summaries every N seconds.
        :param call_request_tensors:
            dictionary `dict{id: Tensor/Op}` containing tensors or operations which should be called
            every N steps or seconds.

            The resulting data will be passed as a `dict{id: result}` to `call_fn`, if specified.
        :param call_fn: callable function which should be called every N steps or seconds.

            This function should accept three arguments. It will be called as:
                `call_fn(
                    session: tf.Session,

                    global_step: int,

                    time_measures: list[float],

                    requested_data: dict{id: result}
                )`

            `time_measures` will contain processor time measures (in seconds) of the session runs since
            `call_fn` was executed the last time.

            `requested_data` will contain an equivalent result of session.run(`call_request_tensors`).
        :param asynchronous: If true, `call_fn` will be executed in a new thread.

            This object will keep track of the thread and make sure it has completed before the main thread has ended.
        """
        self._time_measures = []
        self._threads = []

        self._step_offset = step_offset
        self.call_request_tensors = call_request_tensors
        self.call_fn = call_fn
        self.asynchronous = asynchronous

        self.run_secs = run_secs
        self.run_steps = run_steps

        # self.timer = tf.train.SecondOrStepTimer(every_secs=run_secs,
        #                                         every_steps=run_steps)

    def begin(self):
        self._next_step = None

        self._global_step_tensor = tf.train.get_or_create_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use TimedRunHook.")

    def before_run(self, run_context):
        if self._next_step is None:
            self._next_step = run_context.session.run(self._global_step_tensor) + 1

        requests = {"global_step": self._global_step_tensor}
        if self.shall_request():
            if self.call_request_tensors is not None:
                requests = {**requests, **self.call_request_tensors}

        self._start_time = time.time()

        return tf.train.SessionRunArgs(requests)

    def after_run(self, run_context: tf.train.SessionRunContext, run_values: tf.train.SessionRunValues):
        time_delta = time.time() - self._start_time

        global_step = run_values.results["global_step"]

        if global_step < self._next_step:
            return

        self._time_measures.append(time_delta)

        if self.shall_request():
            if self.call_fn is not None:
                request_data: dict = run_values.results.copy()
                del request_data["global_step"]

                args = (run_context.session, global_step, self._time_measures, request_data)
                if self.asynchronous:
                    thread = threading.Thread(
                        target=self.call_fn,
                        args=args
                    )
                    thread.start()
                    self._threads.append(thread)
                else:
                    self.call_fn(*args)

            self._time_measures.clear()

        self._next_step = global_step + 1

    def shall_request(self):
        if self.run_steps is not None and (self._next_step - self._step_offset) % self.run_steps == 0:
            return True
        if self.run_secs is not None and len(self._time_measures) > 0 and np.sum(self._time_measures) > time.time():
            return True

        return False

    def end(self, session):
        for i in self._threads:
            i.join()


class StopAtLossHook(tf.train.SessionRunHook):
    _global_step_tensor: tf.Tensor

    def __init__(self,
                 loss_tensor: tf.Tensor,
                 min_loss_change=0.001,
                 loss_averaging_steps=50,
                 ):
        """
        Requests stop if the loss change drops below a certain threshold.
        
        :param loss_tensor: The tensor specifying the loss
        :param min_loss_change: minimum change between loss of the last step and the current loss.
        :param loss_averaging_steps:
            Number of steps which will be remembered for averaging.

            E.g. a value of '25' would mean that the loss change at step `i` would be calculated as
                `mean_loss(i-24, [...], i) - mean_loss(i-49, [...], i-25)`.
            Useful in cases where the loss is not monotonously falling, e.g. when using mini-batches.
        """
        self._loss_history = np.repeat(np.inf, loss_averaging_steps)
        self._last_step = None

        self._last_loss = None

        self.loss_tensor: tf.Tensor = loss_tensor

        self.min_loss_change = min_loss_change

    def begin(self):
        self._global_step_tensor = tf.train.get_or_create_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use StopAtStepHook.")

    def after_create_session(self, session, coord):
        if self._last_step is None:
            self._last_step = session.run(self._global_step_tensor)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({
            "global_step": self._global_step_tensor,
            "loss": self.loss_tensor
        })

    def after_run(self, run_context, run_values):
        global_step = run_values.summary["global_step"]
        loss = run_values.summary["loss"]

        if global_step >= self._last_step:
            loss_change = self.loss_change
            if loss_change < self.min_loss_change:
                run_context.request_stop()

            if global_step % self._loss_history.size == 0 and not np.isinf(loss_change):
                tf.logging.info("loss change at step %d: %s" % (global_step, loss_change))

            self._last_step = global_step
            self._update_loss_change(global_step, loss)

    @property
    def loss_change(self):
        if self._last_loss is None:
            return np.inf
        else:
            return np.abs(self._last_loss - np.mean(self._loss_history))

    def _update_loss_change(self, step, loss_value):
        if step % self._loss_history.size == 0:
            self._last_loss = np.mean(self._loss_history)

        self._loss_history[step % self._loss_history.size] = loss_value


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
            {tf.Variable: callable} mappings.
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
                optim_GD = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
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
                optim_Adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
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
                optim_Adagrad = tf.train.AdagradOptimizer(learning_rate=learning_rate)
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
                optim_RMSProp = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
                train_op_RMSProp = optim_RMSProp.apply_gradients(gradients_vars, global_step=global_step)
                if apply_train_ops is not None:
                    train_op_RMSProp = apply_train_ops(train_op_RMSProp)
                update_op_RMSProp = tf.multiply(gradients, learning_rate)  # TODO replace by actual step
            else:
                optim_RMSProp = None
                train_op_RMSProp = None
                update_op_RMSProp = None

            # TFP optimizers:
            #optim_bfgs = None
            #if provide_optimizers["bfgs"]:
            #    logger.debug(" **** Building optimizer: BFGS")
            #    train_op_bfgs = tfp.optimizer.bfgs_minimize(
            #        value_and_gradients_function = (gradients[0], fn),
            #        initial_position,  # TODO: use init here
            #        tolerance=1e-08,
            #        x_tolerance=0,
            #        f_relative_tolerance=0,
            #        initial_inverse_hessian_estimate=None,
            #        max_iterations=50,
            #        parallel_iterations=1
            #    )
            #    # Writes results as namedtuple into train_op_bfgs.
            #else:
            #    train_op_bfgs = None

            # Custom optimizers.
            if provide_optimizers["nr"] and newton_delta is not None:
                logger.debug(" *** Building optimizer: NR")
                update_op_nr = newton_delta

                theta_new_nr = variables - newton_delta
                train_op_nr = tf.group(
                    tf.assign(variables, theta_new_nr),
                    tf.assign_add(global_step, 1)
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
                    tf.assign(variables, theta_new_irls),
                    tf.assign_add(global_step, 1)
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
                    tf.assign(variables, theta_new_irls_gd),
                    tf.assign_add(global_step, 1)
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
                                                        tf.assign_add(global_step, 1))}
                update_op_nr_tr = train_ops_nr_tr["update"]
            else:
                train_op_nr_tr = None
                update_op_nr_tr = None

            if provide_optimizers["irls_tr"] and train_ops_irls_tr is not None:
                logger.debug(" *** Building optimizer: IRLS_TR")
                train_op_irls_tr = {"trial_op": train_ops_irls_tr["trial_op"],
                                    "update_op": tf.group(train_ops_irls_tr["update_op"],
                                                          tf.assign_add(global_step, 1))}
                update_op_irls_tr = train_ops_irls_tr["update"]
            else:
                train_op_irls_tr = None
                update_op_irls_tr = None

            if provide_optimizers["irls_gd_tr"] and train_ops_irls_gd_tr is not None:
                logger.debug(" *** Building optimizer: IRLS_GD_TR")
                train_op_irls_gd_tr = {"trial_op": train_ops_irls_gd_tr["trial_op"],
                                    "update_op": tf.group(train_ops_irls_gd_tr["update_op"],
                                                          tf.assign_add(global_step, 1))}
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
