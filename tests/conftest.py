"""Centralised fixtures and mock configurations testing rocky_uniaxc functionality."""

import json
import os
from pathlib import Path
import sqlite3
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def pytest_configure(config):
    """Run before test collection: mock ansys APIs and resolve any circular imports."""

    # Mock out ansys.rocky.core so tests don't require an actual Rocky DEM installation
    if "ansys" not in sys.modules:
        sys.modules["ansys"] = types.ModuleType("ansys")
    if "ansys.rocky" not in sys.modules:
        sys.modules["ansys.rocky"] = types.ModuleType("ansys.rocky")
    if "ansys.rocky.core" not in sys.modules:
        core_mod = types.ModuleType("ansys.rocky.core")
        core_mod.launch_rocky = lambda *a, **kw: None
        sys.modules["ansys.rocky.core"] = core_mod


# General settings and mesh configurations


@pytest.fixture
def valid_settings_kwargs(tmp_path):
    """Provides bare minimum arguments needed for Settings, skipping real GMSH calls."""
    return {
        "project_dir": str(tmp_path / "test_project"),
        "particle_box_len": 0.01,
        "t_fill": 1.0,
        "t_settle": 0.5,
        "t_compress": 2.0,
        "p_compress": 1000.0,
        "p_radius": 0.001,
        "p_density": 2700,
        "p_poisson": 0.25,
        "p_youngmod": 5e6,
        "fric_dyn_pp": 0.5,
        "fric_stat_pp": 0.3,
        "cor_pp": 0.9,
        "fric_dyn_pw": 0.5,
        "fric_stat_pw": 0.3,
        "cor_pw": 0.9,
    }


@pytest.fixture
def mock_create_meshes():
    """Patch create_meshes to prevent GMSH calls during Settings construction."""
    with (
        patch("rocky_uniaxc.pyrocky.uniax.create_meshes"),
        patch("rocky_uniaxc.compr_meshgen.create_meshes"),
    ):
        yield


# DOE specific setup


@pytest.fixture
def sample_shape_config():
    from rocky_uniaxc.doe._doe_utils import ShapeConfig

    return ShapeConfig(
        name="polyhedron", vert_ar=1.5, horiz_ar=1.2, n_corners=20, sq_degree=3.0
    )


@pytest.fixture
def sample_sim_params(sample_shape_config):
    from rocky_uniaxc.doe._doe_utils import SimParams

    return SimParams(
        radius=0.001,
        density=2700,
        poisson=0.25,
        youngmod=5e6,
        fric_dyn_pp=0.5,
        fric_stat_pp=0.3,
        fric_rolling_pp=0.1,
        cor_pp=0.9,
        fric_dyn_pw=0.5,
        fric_stat_pw=0.3,
        cor_pw=0.9,
        box_len=0.01,
        p_compress=1000.0,
        normal="linear_hysteresis",
        tangential="coulomb_limit",
        rolling="none",
        adhesion="none",
        shape=sample_shape_config,
    )


# Simulates a results.db file produced by a sweep study
@pytest.fixture
def sweep_results_db(tmp_path):
    """Create a minimal results.db in a sweep directory structure."""
    sweep_dir = tmp_path / "test_sweep"
    sweep_dir.mkdir()
    db_path = sweep_dir / "results.db"

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_n INTEGER,
            p_radius REAL,
            bulk_density REAL,
            compressed_density REAL,
            hausner_ratio REAL
        )
    """)
    rows = [
        (1, 0.001, 1500.0, 1800.0, 1.2),
        (2, 0.002, 1600.0, 2000.0, 1.25),
        (3, 0.001, 1400.0, 1650.0, 1.18),
    ]
    cursor.executemany(
        "INSERT INTO results (case_n, p_radius, bulk_density, compressed_density, hausner_ratio) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    return sweep_dir


# Dummy JSON configs to mock inputs
@pytest.fixture
def sweep_json(tmp_path):
    """Write a sweep config JSON to tmp_path and return its path."""
    config = {
        "shape": [
            {
                "name": "sphere",
                "vert_ar": 1,
                "horiz_ar": 1,
                "n_corners": 6,
                "sq_degrees": 2.0,
            }
        ],
        "particle_properties": {
            "radius": [0.001],
            "density": [2700],
            "poisson": [0.25],
            "youngmod": [5e6],
        },
        "inseractions": {
            "pp": {
                "fric_dyn": [0.5],
                "fric_stat": [0.3],
                "fric_rolling": [0.1],
                "cor": [0.9],
            },
            "pw": {
                "fric_dyn": [0.5],
                "fric_stat": [0.3],
                "fric_rolling": [0.1],
                "cor": [0.9],
            },
        },
        "experim_settings": {"box_len": [0.01, 0.02], "p_compress": [1000.0]},
        "contact_model": {
            "normal": ["linear_hysteresis"],
            "tangential": ["coulomb_limit"],
            "rolling": ["none"],
            "adhesion": ["none"],
        },
    }
    path = tmp_path / "sweep_config.json"
    path.write_text(json.dumps(config))
    return str(path)


@pytest.fixture
def ofat_json(tmp_path):
    """Write an OFAT base config JSON and return its path."""
    config = {
        "shape": {
            "name": "polyhedron",
            "vert_ar": 1,
            "horiz_ar": 1,
            "n_corners": 10,
            "sq_degree": 2.0,
        },
        "particle_properties": {
            "radius": 150e-6,
            "density": 2700,
            "poisson": 0.25,
            "youngmod": 5e6,
        },
        "inseractions": {
            "pp": {"fric_dyn": 0.7, "fric_stat": 0.3, "fric_rolling": 0.1, "cor": 0.4},
            "pw": {"fric_dyn": 0.7, "fric_stat": 0.3, "fric_rolling": 0.1, "cor": 0.4},
        },
        "experim_settings": {"box_len": 0.0025, "p_compress": 15e3},
        "contact_model": {
            "normal": "linear_hysteresis",
            "tangential": "coulomb_limit",
            "rolling": "none",
            "adhesion": "none",
        },
    }
    path = tmp_path / "ofat_base.json"
    path.write_text(json.dumps(config))
    return str(path)


# Simulated Rocky environment objects
@pytest.fixture
def mock_rocky_api():
    """Creates a mock object representing an active ansys.rocky.core API session."""
    rocky = MagicMock()
    rocky.api = MagicMock()
    rocky.close = MagicMock()
    return rocky


@pytest.fixture
def fake_rocky_on_path(tmp_path, monkeypatch) -> Path:
    """Create a fake `Rocky` executable and put it on PATH.

    Many parts of the codebase locate Rocky via `shutil.which("Rocky")`. This
    fixture makes that resolution succeed without requiring a real Rocky DEM
    installation.

    Returns:
        Path to the fake Rocky executable.
    """

    bin_dir = tmp_path / "fake_bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    rocky_exe = bin_dir / "Rocky"
    rocky_exe.write_text("#!/bin/sh\nexit 0\n")
    rocky_exe.chmod(0o755)

    monkeypatch.setenv(
        "PATH",
        f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
    )

    return rocky_exe
