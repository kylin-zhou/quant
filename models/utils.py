from typing import Optional, Text, IO, Union
import os

import tempfile
import torch.nn as nn


def get_or_create_path(path: Optional[Text] = None, return_dir: bool = False):
    """Create or get a file or directory given the path and return_dir.
    Parameters
    ----------
    path: a string indicates the path or None indicates creating a temporary path.
    return_dir: if True, create and return a directory; otherwise c&r a file.
    """
    if path:
        if return_dir and not os.path.exists(path):
            os.makedirs(path)
        elif not return_dir:  # return a file, thus we need to create its parent directory
            xpath = os.path.abspath(os.path.join(path, ".."))
            if not os.path.exists(xpath):
                os.makedirs(xpath)
    else:
        temp_dir = os.path.expanduser("~/tmp")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        if return_dir:
            _, path = tempfile.mkdtemp(dir=temp_dir)
        else:
            _, path = tempfile.mkstemp(dir=temp_dir)
    return path


def count_parameters(models_or_parameters, unit="m"):
    """
    This function is to obtain the storage size unit of a (or multiple) models.

    Parameters
    ----------
    models_or_parameters : PyTorch model(s) or a list of parameters.
    unit : the storage size unit.

    Returns
    -------
    The number of parameters of the given model(s) or parameters.
    """
    if isinstance(models_or_parameters, nn.Module):
        counts = sum(v.numel() for v in models_or_parameters.parameters())
    elif isinstance(models_or_parameters, nn.Parameter):
        counts = models_or_parameters.numel()
    elif isinstance(models_or_parameters, (list, tuple)):
        return sum(count_parameters(x, unit) for x in models_or_parameters)
    else:
        counts = sum(v.numel() for v in models_or_parameters)
    unit = unit.lower()
    if unit in ("kb", "k"):
        counts /= 2 ** 10
    elif unit in ("mb", "m"):
        counts /= 2 ** 20
    elif unit in ("gb", "g"):
        counts /= 2 ** 30
    elif unit is not None:
        raise ValueError("Unknown unit: {:}".format(unit))
    return counts



import logging
from typing import Optional, Text, Dict, Any
import re
from logging import config as logging_config

class MetaLogger(type):
    def __new__(mcs, name, bases, attrs):  # pylint: disable=C0204
        wrapper_dict = logging.Logger.__dict__.copy()
        for key in wrapper_dict:
            if key not in attrs and key != "__reduce__":
                attrs[key] = wrapper_dict[key]
        return type.__new__(mcs, name, bases, attrs)


class QlibLogger(metaclass=MetaLogger):
    """
    Customized logger for Qlib.
    """

    def __init__(self, module_name):
        self.module_name = module_name
        # this feature name conflicts with the attribute with Logger
        # rename it to avoid some corner cases that result in comparing `str` and `int`
        self.__level = 0

    @property
    def logger(self):
        logger = logging.getLogger(self.module_name)
        logger.setLevel(self.__level)
        return logger

    def setLevel(self, level):
        self.__level = level

    def __getattr__(self, name):
        # During unpickling, python will call __getattr__. Use this line to avoid maximum recursion error.
        if name in {"__setstate__"}:
            raise AttributeError
        return self.logger.__getattribute__(name)


def get_module_logger(module_name, level: Optional[int] = None) -> QlibLogger:
    """
    Get a logger for a specific module.

    :param module_name: str
        Logic module name.
    :param level: int
    :return: Logger
        Logger object.
    """
    if level is None:
        level = 0

    module_name = "qlib.{}".format(module_name)
    # Get logger.
    module_logger = QlibLogger(module_name)
    module_logger.setLevel(level)
    return module_logger