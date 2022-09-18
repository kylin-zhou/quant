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

def get_logger(output_path=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Create a standard formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    )
    # Create a handler for output to the console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        log_file = os.path.join(output_path, "log.txt")
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger