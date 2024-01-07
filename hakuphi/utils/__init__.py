import importlib
import toml
from inspect import isfunction
from random import shuffle

import torch
import torch.nn as nn


class PickleableTomlDecoder(toml.TomlDecoder):
    def get_empty_inline_table(self):
        return self.get_empty_table()


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate(obj):
    if isinstance(obj, str):
        return get_obj_from_str(obj)
    return obj


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def zero_module(module: nn.Module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def random_choice(
    x: torch.Tensor,
    num: int,
):
    rand_x = list(x)
    shuffle(rand_x)

    return torch.stack(rand_x[:num])


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params
