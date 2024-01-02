import importlib
import inspect
import sys
from dataclasses import dataclass, fields
from inspect import signature
from types import ModuleType
from typing import Any, Callable, cast, Dict, List, Mapping, Optional, TypeVar, Union

from torch import nn

# Taken from torchvision.models._api
M = TypeVar("M", bound=nn.Module)

BUILTIN_MODELS = {}

def register_model(name: Optional[str] = None) -> Callable[[Callable[..., M]], Callable[..., M]]:
    def wrapper(fn: Callable[..., M]) -> Callable[..., M]:
        key = name if name is not None else fn.__name__
        if key in BUILTIN_MODELS:
            raise ValueError(f"An entry is already registered under the name '{key}'.")
        BUILTIN_MODELS[key] = fn
        return fn

    return wrapper