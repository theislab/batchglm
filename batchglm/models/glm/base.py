import abc


class BasicGLM(metaclass=abc.ABCMeta):
    """
    Generalized Linear Model base class.

    Every GLM contains parameters for a location and a scale model
    in a parameter specific linker space and a design matrix for
    each location and scale model.

        - par_link_loc, par_link_scale: Model parameters in linker space.
        - location, scale: Model parameters in output space.
        - link_loc, link_scale: Transform output support to model parameter support.
        - inverse_link_loc: Transform model parameter support to output support.
        - design_loc, design_scale: design matrices
    """

    @property
    @abc.abstractmethod
    def design_loc(self):
        pass

    @property
    @abc.abstractmethod
    def design_scale(self):
        pass

    @property
    @abc.abstractmethod
    def par_link_loc(self):
        pass

    @property
    @abc.abstractmethod
    def par_link_scale(self):
        pass

    @property
    def location(self):
        return self.inverse_link_loc(self.design_loc @ self.par_link_loc)

    @property
    def scale(self):
        return self.inverse_link_scale(self.design_scale @ self.par_link_scale)

    @abc.abstractmethod
    def link_loc(self, data):
        pass

    @abc.abstractmethod
    def link_scale(self, data):
        pass

    @abc.abstractmethod
    def inverse_link_loc(self, data):
        pass

    @abc.abstractmethod
    def inverse_link_scale(self, data):
        pass

