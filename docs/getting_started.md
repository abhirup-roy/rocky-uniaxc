# Getting Started

This guide helps you install and configure **Rocky-UniaxC** for multiscale uniaxial compression simulations in Ansys Rocky.

## Prerequisites

Before installing the package, make sure you meet the following requirements:

*   **Python:** Version 3.10, 3.11, 3.12, or 3.13.
*   **Ansys Rocky DEM:** Ansys Rocky version 2025 R2 (or compatible release) with its Python API (`ansys-rocky-core`) installed.
*   **GMSH:** Required for generating STL meshes of walls and insert surfaces.
*   **SLURM** (Optional): A SLURM-managed HPC cluster is required if you plan to launch sweeps using the scheduler module.

## Installation

You can install the package directly using `pip` or [uv](https://github.com/astral-sh/uv).

### Using `uv` (Recommended)

Sync development and testing dependencies in a virtual environment:

```bash
uv sync --all-extras --dev
```

Install the package into your current python environment:

```bash
uv pip install .
```

### Using `pip`

Install standard package:

```bash
pip install .
```

Install with testing dependencies:

```bash
pip install .[test]
```

Install with documenting dependencies:

```bash
pip install .[docs]
```

## Verification

To verify the installation and run the unit test suite, execute:

```bash
uv run pytest
```

## Configuring the Rocky Executable Path

Rocky-UniaxC attempts to auto-detect the Rocky executable path on import. If Rocky is installed in a non-standard location, you can configure it programmatically:

```python
import rocky_uniaxc

rocky_uniaxc.set_rocky_exe_path("/path/to/Ansys/Rocky/bin/Rocky")
```

By default, simulations run in headless mode. You can toggle this setting using:

```python
rocky_uniaxc.set_headless_mode(False)  # Run with Rocky GUI visible
```
