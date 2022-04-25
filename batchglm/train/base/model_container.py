import abc
from ...models.base_glm import ModelGLM

class BaseModelContainer(metaclass=abc.ABCMeta):

        @abc.abstractmethod
        def error_codes(self):
            pass

        @abc.abstractmethod
        def niter(self):
            pass

        @abc.abstractmethod
        def loss(self):
            pass

        @abc.abstractmethod
        def log_likelihood(self):
            pass

        @abc.abstractmethod
        def jacobian(self):
            pass

        @abc.abstractmethod
        def hessian(self):
            pass

        @abc.abstractmethod
        def fisher_inv(self):
            pass

        @abc.abstractmethod
        def x(self):
            pass

        @abc.abstractmethod
        def model(self):
            pass

        @abc.abstractmethod
        def theta_location(self):
            pass

        @abc.abstractmethod
        def model(self) -> ModelGLM:
            pass

