from typing import Union, Dict, Callable, List

import time
import threading

import tensorflow as tf
import numpy as np


# class StepTimeMonitorHook(tf.train.SessionRunHook):
#     """Measures the time needed for every step"""
#
#     measures: list
#     _start_time: float
#
#     def __init__(self):
#         self.measures = []
#
#     def before_run(self, run_context):
#         self._start_time = time.time()
#
#     def after_run(self, run_context: tf.train.SessionRunContext, run_values: tf.train.SessionRunValues):
#
#     def __str__(self):
#         return "%s: measures=%s" % (self.__class__, str(self.measures))
#
#     def __repr__(self):
#         return self.__str__()


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
        global_step = run_values.results["global_step"]
        loss = run_values.results["loss"]

        if global_step >= self._last_step:
            if self.loss_change < self.min_loss_change:
                run_context.request_stop()

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
