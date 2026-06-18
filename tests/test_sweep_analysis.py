"""Validate data extraction and collation tools for post-processing Rocky results."""

import os
from unittest.mock import patch
import pandas as pd
import pytest
import rocky_uniaxc.sweep_analysis as sa


class TestLoadData:
    def test_load_data_valid(self, sweep_results_db):
        with patch.object(sa, "PWD", str(sweep_results_db.parent)):
            df = sa.load_data("test_sweep")
        assert len(df) == 3
        assert "p_radius" in df.columns

    def test_load_data_missing_db(self, tmp_path):
        with patch.object(sa, "PWD", str(tmp_path)):
            with pytest.raises(FileNotFoundError, match="Results db not found"):
                sa.load_data("nonexistent_sweep")


class TestDumpResults:
    def test_dump_csv(self, sweep_results_db):
        with patch.object(sa, "PWD", str(sweep_results_db.parent)):
            sa.dump_results("test_sweep", filetype="csv")
            output_file = sweep_results_db / "pyoutputs" / "results.csv"
            assert output_file.exists()

    def test_dump_minimal(self, sweep_results_db):
        with patch.object(sa, "PWD", str(sweep_results_db.parent)):
            sa.dump_results("test_sweep", filetype="csv", minimal=True)
            output_file = sweep_results_db / "pyoutputs" / "results.csv"
            assert output_file.exists()
            df = pd.read_csv(output_file)
            # minimal drops constant columns; hausner_ratio varies (1.2, 1.25, 1.18)
            assert "hausner_ratio" in df.columns


class TestFindFaultyRuns:
    def _make_case_dir(self, sweep_dir, case_name, hausner_ratio):
        case_path = sweep_dir / case_name
        case_path.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([{"hausner_ratio": hausner_ratio, "p_radius": 0.001}])
        df.to_csv(case_path / "results.csv", index=False)
        return case_path

    def test_no_faulty(self, tmp_path, capsys):
        sweep_dir = tmp_path / "sweep_ok"
        sweep_dir.mkdir()
        self._make_case_dir(sweep_dir, "case_0", 1.2)
        self._make_case_dir(sweep_dir, "case_1", 1.5)
        with patch.object(sa, "PWD", str(tmp_path)):
            sa.find_faulty_runs("sweep_ok")
        captured = capsys.readouterr()
        assert "No faulty cases found" in captured.out

    def test_with_faulty(self, tmp_path, capsys):
        sweep_dir = tmp_path / "sweep_bad"
        sweep_dir.mkdir()
        self._make_case_dir(sweep_dir, "case_0", 0.5)  # < 1 -> faulty
        self._make_case_dir(sweep_dir, "case_1", 1.2)
        with patch.object(sa, "PWD", str(tmp_path)):
            sa.find_faulty_runs("sweep_bad")
        captured = capsys.readouterr()
        assert "case_0" in captured.out

    def test_dump_faulty(self, tmp_path):
        sweep_dir = tmp_path / "sweep_dump"
        sweep_dir.mkdir()
        self._make_case_dir(sweep_dir, "case_0", 0.5)
        with patch.object(sa, "PWD", str(tmp_path)):
            sa.find_faulty_runs("sweep_dump", dump=True)
        assert (sweep_dir / "faulty_cases.txt").exists()


class TestDumpResultsBackup:
    def test_combines_case_csvs(self, tmp_path):
        sweep_dir = tmp_path / "sweep_backup"
        sweep_dir.mkdir()
        for i in range(3):
            case_path = sweep_dir / f"case_{i}"
            case_path.mkdir()
            df = pd.DataFrame([{"case_n": i, "p_radius": 0.001 * (i + 1)}])
            df.to_csv(case_path / "results.csv", index=False)
        with patch.object(sa, "PWD", str(tmp_path)):
            sa.dump_results_backup("sweep_backup", filetype="csv")
        output = sweep_dir / "all_results.csv"
        assert output.exists()
        df = pd.read_csv(output)
        assert len(df) == 3

    def test_missing_dir(self, tmp_path):
        with patch.object(sa, "PWD", str(tmp_path)):
            with pytest.raises(NotADirectoryError, match="Sweep directory not found"):
                sa.dump_results_backup("nonexistent")
