"""Validate RockyScheduler script generation and submission."""

import subprocess
from unittest.mock import patch

import pytest

from rocky_uniaxc.utils import RockyScheduler


class TestGenerate:
    def test_bb_cpu(self, tmp_path):
        path = RockyScheduler.bb_cpu(ncpus=20).generate(tmp_path)
        assert path.name == "runRocky.sh"
        script = path.read_text()
        assert "#!/bin/bash" in script
        assert "--ntasks=20" in script
        assert "--qos=bbdefault" in script
        assert "Rocky --script" in script

    def test_bb_cpu_default_ncpus(self, tmp_path):
        script = RockyScheduler.bb_cpu().generate(tmp_path).read_text()
        assert "--ntasks=20" in script

    def test_az_gpu(self, tmp_path):
        script = RockyScheduler.az_gpu(ngpus=2).generate(tmp_path).read_text()
        assert "#!/bin/sh" in script
        assert "--gres=gpu:2" in script
        assert "--partition=long-gpu" in script

    def test_bb_gpu(self, tmp_path):
        script = RockyScheduler.bb_gpu(ngpus=1).generate(tmp_path).read_text()
        assert "--qos=bbgpu" in script
        assert "--gres=gpu:1" in script

    def test_run_days(self, tmp_path):
        script = (
            RockyScheduler.bb_cpu(ncpus=20, run_days=5).generate(tmp_path).read_text()
        )
        assert "--time=5-0" in script

    def test_extra_directive(self, tmp_path):
        script = RockyScheduler(constraint="cascadelake").generate(tmp_path).read_text()
        assert "--constraint=cascadelake" in script

    def test_none_directives_omitted(self, tmp_path):
        script = RockyScheduler(qos=None, gres=None).generate(tmp_path).read_text()
        assert "--qos" not in script
        assert "--gres" not in script


class TestCustom:
    def test_custom_verbatim(self, tmp_path):
        text = "#!/bin/bash\necho hello"
        script = RockyScheduler.custom(text).generate(tmp_path).read_text()
        assert script == text

    def test_custom_invalid_shebang(self):
        with pytest.raises(ValueError, match="shebang"):
            RockyScheduler.custom("not bash")


class TestLaunch:
    @patch("subprocess.run")
    def test_submit_calls_sbatch(self, mock_run, tmp_path):
        scheduler = RockyScheduler.bb_cpu()
        scheduler.generate(tmp_path)
        scheduler.submit(tmp_path)
        mock_run.assert_called_once()
        assert "sbatch" in mock_run.call_args[0][0]

    @patch("subprocess.run")
    def test_launch_all_success(self, mock_run, tmp_path):
        dirs = [tmp_path / "case_0", tmp_path / "case_1"]
        for d in dirs:
            d.mkdir()
        failed = RockyScheduler.bb_cpu().launch_all(dirs)
        assert failed == []
        assert mock_run.call_count == 2

    @patch("subprocess.run")
    def test_launch_all_collects_failures(self, mock_run, tmp_path):
        dirs = [tmp_path / "case_0", tmp_path / "case_1"]
        for d in dirs:
            d.mkdir()
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, "sbatch", stderr="boom"),
            None,
        ]
        failed = RockyScheduler.bb_cpu().launch_all(dirs)
        assert failed == [0]
