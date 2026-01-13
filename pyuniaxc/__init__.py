"""
pyuniaxc: Tools for setting up multiscale uniaxial compression

This package provides tools for setting up and analysing
multiscale uniaxial compression.

Author: Abhirup Roy
"""

__version__ = "0.1"
__author__ = "Abhirup Roy"
__all__ = ["launch_sweep", "launch_ofat", "analyse", "externals"]

from doe.sweep import launch_sweep
from doe.ofat import launch_ofat
from . import sweep_analysis as analyse
from . import externals
