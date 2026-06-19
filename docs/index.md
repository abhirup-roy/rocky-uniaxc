# Rocky-UniaxC: Multiscale Uniaxial Compression Simulation Toolkit for Ansys Rocky

[![Build and Test](https://github.com/abhirup-roy/rocky-uniaxc/actions/workflows/ci.yml/badge.svg)](https://github.com/abhirup-roy/rocky-uniaxc/actions/workflows/ci.yml)
![Python Versions](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?logo=python)

**Rocky-UniaxC** is a Python toolkit for configuring, launching, and post-processing multiscale uniaxial compression simulations using Ansys Rocky DEM.

It provides support for:
*   **Design of Experiments (DOE):** Full sweeps and One-Factor-at-a-Time (OFAT) parameter designs.
*   **Automatic Mesh Generation:** boundary wall and insert surface geometry via GMSH.
*   **HPC Schedulers:** Presets for SLURM scheduling scripts (BlueBear and SCP cluster support).
*   **Analysis & Diagnostics:** Processing database tables, logging particle status, and exporting datasets (Pandas, Parquet, CSV, Excel).
