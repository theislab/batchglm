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
    def ll(self):
        pass

    @abc.abstractmethod
    def jac(self):
        pass

    @abc.abstractmethod
    def hessian(self):
        pass

    @abc.abstractmethod
    def fisher_inv(self):
        pass

    @property
    @abc.abstractmethod
    def theta_location(self):
        pass

    @property
    @abc.abstractmethod
    def model(self) -> ModelGLM:
        pass

    def theta_location_constrained(self):
        pass

    def theta_scale_constrained(self):
        pass
