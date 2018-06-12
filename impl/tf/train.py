from typing import Union, Dict, Callable

import tensorflow as tf
import numpy as np


class TimedRunHook(tf.train.SessionRunHook):
    """Runs ops or functions every N steps or seconds."""

    _next_step: int
    _global_step_tensor: tf.Tensor
    _shall_request: bool

    def __init__(self,
                 run_steps=None,
                 run_secs=None,
                 call_request_tensors: Dict[str, tf.Tensor] = None,
                 call_fn: Callable = None
                 ):
        """Initializes a `TimedRunHook`.
    
        Args:
            run_steps: `int`, run summaries every N steps. Exactly one of
              `run_secs` and `run_steps` should be set.
            run_secs: `int`, run summaries every N seconds.
            call_request_tensors:
                dictionary `dict{id: Tensor/Op}` containing tensors or operations which should be called
                every N steps or seconds.
                
                The resulting data will be passed as a `dict{id: result}` to `call_fn`, if specified.
            call_fn:
                callable function which should be called every N steps or seconds.
                
                This function should accept three arguments. It will be called as
                `call_fn(session: tf.Session, global_step: int, requested_data: dict{id: result}).
                
                `requested_data` will contain an equivalent result of session.run(`call_request_tensors`).
    
        """

        self.call_request_tensors = call_request_tensors
        self.call_fn = call_fn

        self.timer = tf.train.SecondOrStepTimer(every_secs=run_secs,
                                                every_steps=run_steps)

    def begin(self):
        self._next_step = None
        self._global_step_tensor = tf.train.get_or_create_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use TimedRunHook.")

    def before_run(self, run_context):  # pylint: disable=unused-argument
        self._shall_request = (
                self._next_step is None or
                self.timer.should_trigger_for_step(self._next_step))
        requests = {"global_step": self._global_step_tensor}
        if self._shall_request:
            if self.call_request_tensors is not None:
                requests = {**requests, **self.call_request_tensors}

        return tf.train.SessionRunArgs(requests)

    def after_run(self, run_context: tf.train.SessionRunContext, run_values: tf.train.SessionRunValues):
        stale_global_step = run_values.results["global_step"]
        global_step = stale_global_step + 1

        if self._next_step is None or self._shall_request:
            global_step = run_context.session.run(self._global_step_tensor)

        if self._shall_request:
            self.timer.update_last_triggered_step(global_step)

            request_data: dict = run_values.results.copy()
            del request_data["global_step"]

            if self.call_fn is not None:
                self.call_fn(run_context.session, global_step, request_data)

        self._next_step = global_step + 1


class StopAtLossHook(tf.train.SessionRunHook):

    def __init__(self,
                 loss_tensor: tf.Tensor,
                 min_loss_change=0.001,
                 loss_averaging_steps=25,
                 ):
        """
        Requests stop if the loss change drops below a certain threshold.
        
        :param loss_tensor: The tensor specifying the loss
        :param min_loss_change: minimum change between last loss and the current loss.
        :param loss_averaging_steps:
            Number of steps which will be remembered for averaging.

            E.g. a value of '25' would mean that the loss change at step `i` would be calculated as
                `mean_loss(i-24, [...], i) - mean_loss(i-49, [...], i-25)`.
            Useful in cases where the loss is not monotonously falling, e.g. when using mini-batches.
        """
        self._last_step = None
        self._global_step_tensor = None

        self._last_loss = None
        self._loss_history = np.repeat(np.inf, loss_averaging_steps)

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
        global_step = run_values.results["global_step"]
        loss = run_values.results["loss"]

        if global_step >= self._last_step:
            self._update_loss_change(global_step, loss)

            if (global_step + 1) % self._loss_history.size == 0:
                loss_change = self.loss_change

                tf.logging.info("loss change at step %d: %s" % (global_step, loss_change))

                if loss_change < self.min_loss_change:
                    run_context.request_stop()

            self._last_step = global_step

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
