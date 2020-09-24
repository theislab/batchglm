import abc
from typing import Union, Any, Dict, Iterable
import logging

from .external import types as T
from .input import InputDataBase

logger = logging.getLogger(__name__)


class _ModelBase(metaclass=abc.ABCMeta):
    r"""
    Model base class
    """

    def __init__(
            self,
            input_data: InputDataBase
    ):
        self.input_data = input_data

    @property
    def x(self) -> T.ArrayLike:
        return self.input_data.x

    @property
    def w(self) -> T.ArrayLike:
        return self.input_data.w

    def get(self, key: Union[str, Iterable]) -> Union[Any, Dict[str, Any]]:
        """
        Returns the values specified by key.

        :param key: Either a string or an iterable list/set/tuple/etc. of strings
        :return: Single array if `key` is a string or a dict {k: value} of arrays if `key` is a collection of strings
        """
        if isinstance(key, str):
            return self.__getattribute__(key)
        elif isinstance(key, Iterable):
            return {s: self.__getattribute__(s) for s in key}

    def __getitem__(self, item) -> Union[Any, Dict[str, Any]]:
        return self.get(item)

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.__str__()
