from .external import ProcessModelBase
import abc


class ProcessModelGLM(ProcessModelBase):

    @abc.abstractmethod
    def param_bounds(self, dtype):
        pass
