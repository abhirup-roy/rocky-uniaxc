"""Validates the matrix generation and slurm submission pipelines for parameter sweeps."""

import json
from pathlib import Path
from unittest.mock import patch
from rocky_uniaxc import RockyScheduler
import pytest

from rocky_uniaxc.doe.sweep import launch_sweep


class TestLaunchSweep:
    def test_launch_sweep_minimal(self, tmp_path, sweep_json):
        sweep_name = "test_launch"

        with patch("rocky_uniaxc.utils.RockyScheduler.generate") as mock_generate:
            with patch("rocky_uniaxc.doe.sweep.create_meshes") as mock_meshes:
                launch_sweep(scheduler=RockyScheduler.bb_cpu(),
                    sweep_name=str(tmp_path / sweep_name),
                    json_path=sweep_json,
                    autolaunch=False,
                    backend="pyrocky",
                )

        # With the dummy sweep_json, it produces 2 combinations
        # (because box_len=[0.01, 0.02] natively, everything else scalar usually)
        sweep_dir = tmp_path / sweep_name
        assert sweep_dir.exists()

        # Ensure meshes are requested appropriately
        assert mock_meshes.called

        # Verify case directory generations
        case_0_json = sweep_dir / "case_0" / "settings.json"
        case_1_json = sweep_dir / "case_1" / "settings.json"

        assert case_0_json.exists()
        assert case_1_json.exists()

        with open(case_0_json, "r") as f:
            data = json.load(f)
            assert "p_radius" in data

        assert sweep_dir.joinpath("case_0", "script_uniax.py").exists()

        # Verify it generates a submission script per case
        assert mock_generate.call_count == 2
