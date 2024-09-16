# -*- coding: utf-8 -*-
"""
rosenpy package initialization file.

This file initializes the `rosenpy` package and exposes the main submodules for external use.
"""

# rosenpy/__init__.py

# Exposing the main submodules
from . import model  # The model submodule contains neural network models and architectures.
from . import utils  # The utils submodule contains utility functions like activation functions and initialization.

__all__ = ['model', 'utils']
"""
__all__ specifies the public API of the `rosenpy` package.

Attributes:
-----------
model : module
    Contains neural network models, including architectures and training routines.
utils : module
    Provides utility functions, such as activation functions, weight initialization, and other helper methods.
"""
