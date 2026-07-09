"""Tools for setting up and analysing multiscale uniaxial compression simulations.

This package provides utilities for configuring, launching, and post-processing
uniaxial compression experiments using Ansys Rocky DEM. It supports full
parameter sweeps, one-factor-at-a-time (OFAT) designs, and pyrocky-based
simulation workflows.

Author:
    Abhirup Roy
"""

__version__ = "0.1"
__author__ = "Abhirup Roy"
__all__ = [
    "launch_sweep",
    "launch_ofat",
    "launch_space_filling",
    "analyse",
    "externals",
    "pyrocky",
]


HEADLESS = True
BACKEND = "pyrocky"
ROCKY_EXE_PATH = None

import pathlib as _pathlib
from .doe.sweep import launch_sweep
from .doe.ofat import launch_ofat
from .doe.space_filling import launch_space_filling
from .utils import RockyScheduler
# from .doe import med
from . import sweep_analysis as analyse
from . import externals
from . import pyrocky

# Auto-detect Rocky executable path at import time
ROCKY_EXE_PATH = pyrocky.find_rocky_exe()


def set_rocky_exe_path(path: str) -> None:
    """Set the path to the Rocky executable for the rocky_uniaxc package.

    Allows users to specify the path to the Rocky executable if it is not in a
    standard location or not found automatically. The pyrocky API will use the
    specified executable for running simulations.

    Args:
        path: The file path to the Rocky executable.

    Raises:
        FileNotFoundError: If the specified path does not point to a valid file.

    Example:
        >>> set_rocky_exe_path("/path/to/rocky/executable")
    """

    if not _pathlib.Path(path).is_file():
        raise FileNotFoundError(f"Specified Rocky executable not found at: {path}")

    global ROCKY_EXE_PATH
    ROCKY_EXE_PATH = path


def set_headless_mode(headless: bool) -> None:
    """Set the headless mode for Rocky simulations.

    Controls whether Rocky simulations run in headless mode (without a GUI) or
    with the graphical interface. This setting affects how the pyrocky API
    launches Rocky.

    Args:
        headless: If ``True``, Rocky runs in headless mode. If ``False``,
            Rocky launches with its GUI.

    Example:
        >>> set_headless_mode(True)   # batch processing / servers
        >>> set_headless_mode(False)  # interactive use / debugging
    """
    global HEADLESS
    HEADLESS = headless


def set_backend(backend: str) -> None:
    """Set the backend for Rocky simulations.

    Determines how the package interacts with the Rocky executable.
    All non-simulation utilities use pyrocky regardless of this setting.

    Args:
        backend: The backend to use. Must be ``"pyrocky"`` or
            ``"rocky_prepost"``.

    Raises:
        ValueError: If an unsupported backend is specified.

    Example:
        >>> set_backend("pyrocky")
    """
    if backend not in ["pyrocky", "rocky_prepost"]:
        raise ValueError(
            f"Unsupported backend: {backend}. Supported backends are 'pyrocky' and 'rocky_prepost'."
        )
    global BACKEND
    BACKEND = backend
