from typing import Union, Dict, Callable

import tensorflow as tf


class TimedRunHook(tf.train.SessionRunHook):
    """Runs ops or functions every N steps or seconds."""
    
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
        
        self._timer = tf.train.SecondOrStepTimer(every_secs=run_secs,
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
                self._timer.should_trigger_for_step(self._next_step))
        requests = {"global_step": self._global_step_tensor}
        if self._shall_request:
            if self._request is not None:
                requests = {**requests, **self._request}
        
        return tf.train.SessionRunArgs(requests)
    
    def after_run(self, run_context: tf.train.SessionRunContext, run_values: tf.train.SessionRunValues):
        stale_global_step = run_values.results["global_step"]
        global_step = stale_global_step + 1
        
        if self._next_step is None or self._shall_request:
            global_step = run_context.session.run(self._global_step_tensor)
        
        if self._shall_request:
            self._timer.update_last_triggered_step(global_step)
            
            request_data: dict = run_values.results.copy()
            del request_data["global_step"]
            
            if self.call_fn is not None:
                self.call_fn(run_context.session, global_step, request_data)
        
        self._next_step = global_step + 1
