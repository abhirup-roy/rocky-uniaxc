# Rocky-UniaxC: Multiscale Uniaxial Compression Simulation Toolkit for Ansys Rocky

[![Build and Test](https://github.com/abhirup-roy/rocky-uniaxc/actions/workflows/ci.yml/badge.svg)](https://github.com/abhirup-roy/rocky-uniaxc/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/rocky-uniaxc.svg)](https://pypi.org/project/rocky-uniaxc/)
![Python Versions](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?logo=python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`rocky-uniaxc` is a Python toolkit designed to automate the configuration, mesh generation, execution, and analysis of multiscale uniaxial compression simulations in **Ansys Rocky DEM**. It supports running simulations locally or deploying parallel job arrays on HPC clusters (like BlueBear or SCP) using SLURM.

---

## 🌟 Key Features

* **Automated Boundary Mesh Generation:** Programmatically creates custom compression wall and insert surfaces using GMSH.
* **Design of Experiments (DOE):**
  * **Full-Factorial Parameter Sweeps:** Automatically expands all parameter combinations to perform comprehensive grid sweeps.
  * **One-Factor-at-a-Time (OFAT):** Evaluates parameter sensitivities by varying one variable at a time while holding others constant.
  * ...more to be added
* **Flexible Simulation Backends:** Works with the modern `pyrocky` API wrapper or standard `rocky_prepost` scripts.
* **SLURM HPC Cluster Presets:** Ready-made presets for submitting parallel simulation batch jobs on UoB BlueBear (CPU/GPU) and SCP (GPU).
* **Robust Post-Processing:** Automated analysis of settled and compressed states (e.g., calculating bulk densities, Coordination/Contact Numbers, and contacts ratio).
* **Data Export & Quality Audits:** Automatically scans logs for lost particles, flags faulty runs, and gathers results into structured databases (SQLite) and dataframes (Pandas/CSV/Parquet).

---

## 📋 Prerequisites

* **Python:** `>= 3.10`
* **Ansys Rocky DEM:** Ansys Rocky version 2025 R2 (or compatible release) with its Python API (`ansys-rocky-core`) installed.
* **GMSH:** Required on your system for generating boundary STL meshes.

---

## ⚙️ Installation

### From PyPI

Install the latest release directly from PyPI:

```bash
pip install rocky-uniaxc
```

### From Source

You can also install from source using `uv` (recommended) or `pip`:

#### Using `uv`

To sync development and testing dependencies in a virtual environment:

```bash
uv sync --all-extras --dev
```

To install the package into your current Python environment:

```bash
uv pip install .
```

#### Using `pip`

Install the core package:

```bash
pip install .
```

Install with testing dependencies:

```bash
pip install .[test]
```

Install with documentation dependencies:

```bash
pip install .[docs]
```

---

## 🚀 Quickstart

### 1. Run a Parameter Sweep

Define your parameter space in a JSON config file (e.g., `sweep_config.json`):

```json
{
  "shape": {
    "name": "sphere"
  },
  "particle_properties": {
    "radius": [0.005, 0.006],
    "density": [2500],
    "poisson": [0.25],
    "youngmod": [1e7]
  },
  "inseractions": {
    "pp": {
      "fric_dyn": [0.3, 0.5],
      "fric_stat": [0.4],
      "fric_rolling": [0.1],
      "cor": [0.5]
    },
    "pw": {
      "fric_dyn": [0.4],
      "fric_stat": [0.5],
      "cor": [0.5]
    }
  },
  "experim_settings": {
    "box_len": [0.1],
    "p_compress": [1000]
  },
  "contact_model": {
    "normal": "linear_hysteresis",
    "tangential": "coulomb_limit",
    "rolling": "none",
    "adhesion": "none"
  }
}
```

Write a script to generate and schedule the simulations:

```python
from rocky_uniaxc import launch_sweep
from rocky_uniaxc.utils import RockyScheduler

# 1. Define the cluster scheduler settings (e.g., BlueBear CPU)
scheduler = RockyScheduler.bb_cpu(ncpus=20, run_days=3)

# 2. Launch the sweep configuration
launch_sweep(
    sweep_name="my_first_sweep",
    scheduler=scheduler,
    json_path="sweep_config.json",
    autolaunch=True,
    target="CPU",
    backend="pyrocky"
)
```

### 2. Post-Process Simulation Results

Load all simulation results from SQLite databases and export them for downstream analysis:

```python
import rocky_uniaxc.sweep_analysis as analyze

# Load data into a Pandas DataFrame
df = analyze.load_data("my_first_sweep")
print(df.head())

# Check for runs with particle loss warnings or outliers
analyze.find_faulty_runs("my_first_sweep", dump=True)

# Export the results in custom formats (parquet, csv, excel)
analyze.dump_results("my_first_sweep", filetype="parquet", minimal=True)
```

### 3. Run a Single Case via CLI

You can execute a single simulation case using a local `settings.json` file:

```bash
python -m rocky_uniaxc.case_runner path/to/settings.json
```

---

## 🧪 Running Tests

To run the unit test suite and check code correctness:

```bash
uv run pytest
```

---

## 📚 Documentation

The documentation is configured for hosting on [ReadTheDocs](https://readthedocs.org/).

To build the HTML documentation locally:

```bash
uv run mkdocs build
```

The compiled output will be available at `site/index.html`.

To run a local live-reload server for development:

```bash
uv run mkdocs serve
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
