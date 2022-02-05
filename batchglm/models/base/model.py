import abc
import logging
from typing import Any, Dict, Iterable, Union, Optional

try:
    import anndata
except ImportError:
    anndata = None

from .input import InputDataBase


logger = logging.getLogger(__name__)


class _ModelBase(metaclass=abc.ABCMeta):
    r"""
    Model base class
    """

    def __init__(self, input_data: Optional[InputDataBase] = None):
        self.input_data = input_data

    @property
    def x(self):
        return self.input_data.x

    def get(self, key: Union[str, Iterable]) -> Union[Any, Dict[str, Any]]:
        """
        Returns the values specified by key.

        :param key: Either a string or an iterable list/set/tuple/etc. of strings
        :return: Single array if `key` is a string or a dict {k: value} of arrays if `key` is a collection of strings
        """
        if isinstance(key, str):
            attrib = self.__getattribute__(key)
        elif isinstance(key, Iterable):
            attrib = {s: self.__getattribute__(s) for s in key}
        return attrib

    def __getitem__(self, item):
        return self.get(item)

    def __repr__(self):
        return self.__str__()
