#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Author  :   Jeremy Bleyer, Ecole Nationale des Ponts et Chaussées, Navier
@Contact :   jeremy.bleyer@enpc.fr
@Time    :   26/08/2024
"""
__version__ = "0.4.0"

# Define PerformanceWarning BEFORE importing submodules to avoid circular import
class PerformanceWarning(UserWarning):
    """Warning for performance-related issues in dolfinx_materials."""
    pass

# IMPORTANT: Add to module globals immediately
import sys
sys.modules[__name__].PerformanceWarning = PerformanceWarning

__all__ = ['PerformanceWarning']

from . import jaxmat
from . import generic
from . import solvers
from . import utils