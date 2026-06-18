"""Validate generic shell macros like directory swapping and sbatch builders."""

import os
from unittest.mock import patch
import pytest
from rocky_uniaxc.utils import cd, slurm_sbatch


class TestCd:
    def test_cd_changes_dir(self, tmp_path):
        original = os.getcwd()
        target = str(tmp_path / "subdir")
        os.makedirs(target)
        with cd(target):
            assert os.getcwd() == target
        assert os.getcwd() == original

    def test_cd_expanduser(self):
        original = os.getcwd()
        with cd("~"):
            assert os.getcwd() == os.path.expanduser("~")
        assert os.getcwd() == original

    def test_cd_nested(self, tmp_path):
        original = os.getcwd()
        dir_a = str(tmp_path / "a")
        dir_b = str(tmp_path / "b")
        os.makedirs(dir_a)
        os.makedirs(dir_b)
        with cd(dir_a):
            assert os.getcwd() == dir_a
            with cd(dir_b):
                assert os.getcwd() == dir_b
            assert os.getcwd() == dir_a
        assert os.getcwd() == original


class TestSlurmSbatch:
    def test_bb_cpu(self, tmp_path):
        slurm_sbatch(str(tmp_path), loc="bb-cpu", ncpus=20)
        script = (tmp_path / "runRocky.sh").read_text()
        assert "#!/bin/bash" in script
        assert "--ntasks=20" in script
        assert "Rocky --script" in script

    def test_bb_cpu_default_ncpus(self, tmp_path):
        slurm_sbatch(str(tmp_path), loc="bb-cpu")
        script = (tmp_path / "runRocky.sh").read_text()
        assert "--ntasks=20" in script

    def test_az_gpu(self, tmp_path):
        slurm_sbatch(str(tmp_path), loc="az-gpu", ngpus=2)
        script = (tmp_path / "runRocky.sh").read_text()
        assert "gres=gpu:2" in script

    def test_bb_gpu(self, tmp_path):
        slurm_sbatch(str(tmp_path), loc="bb-gpu", ngpus=1)
        script = (tmp_path / "runRocky.sh").read_text()
        assert "bbgpu" in script

    def test_custom_valid(self, tmp_path):
        custom_msg = "#!/bin/bash\necho hello"
        slurm_sbatch(str(tmp_path), loc="custom", custom_msg=custom_msg)
        script = (tmp_path / "runRocky.sh").read_text()
        assert "echo hello" in script

    def test_custom_invalid(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid custom message"):
            slurm_sbatch(str(tmp_path), loc="custom", custom_msg="not bash")

    def test_invalid_loc(self, tmp_path):
        with pytest.raises(ValueError, match="Only 'bb-cpu'"):
            slurm_sbatch(str(tmp_path), loc="invalid_loc")

    def test_run_days(self, tmp_path):
        slurm_sbatch(str(tmp_path), loc="bb-cpu", ncpus=20, run_days=5)
        script = (tmp_path / "runRocky.sh").read_text()
        assert "--time=5-0" in script

    @patch("subprocess.run")
    def test_autolaunch_calls_sbatch(self, mock_run, tmp_path):
        slurm_sbatch(str(tmp_path), loc="bb-cpu", ncpus=20, autolaunch=True)
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert "sbatch" in call_args[0][0]
