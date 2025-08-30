"""
pyuniaxc: Tools for setting up multiscale uniaxial compression

This package provides tools for setting up and analysing
multiscale uniaxial compression.

Author: Abhirup Roy
"""

__version__ = "0.1"
__author__ = "Abhirup Roy"
__all__ = ["launch_sweep", "launch_ofat", "analyse"]

from .rocky_sweep import make_cases as launch_sweep
from .rocky_doe import launch_ofat
from . import sweep_analysis as analyse
