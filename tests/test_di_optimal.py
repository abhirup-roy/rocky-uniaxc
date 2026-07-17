"""Tests for D- and I-optimal DOE augmentation."""

import json
from unittest.mock import patch

import pytest

from rocky_uniaxc.doe.di_optimal import iter_di_optimal, launch_di_optimal
from rocky_uniaxc.utils import RockyScheduler


@pytest.mark.parametrize("criterion", ["D", "I"])
def test_iter_di_optimal_augments_existing_ofat(ofat_json, criterion):
    existing = [
        {"cor_pp": 0.1, "fric_dyn_pp": 0.5},
        {"cor_pp": 0.5, "fric_dyn_pp": 0.5},
        {"cor_pp": 0.9, "fric_dyn_pp": 0.5},
        {"cor_pp": 0.5, "fric_dyn_pp": 0.2},
        {"cor_pp": 0.5, "fric_dyn_pp": 0.8},
    ]

    params = iter_di_optimal(
        ofat_json,
        existing,
        factors=["cor_pp", "fric_dyn_pp"],
        bounds=[(0.1, 0.9), (0.2, 0.8)],
        n_samples=3,
        criterion=criterion,
        n_candidates=128,
        seed=7,
    )

    assert len(params) == 3
    assert all(0.1 <= run.cor_pp <= 0.9 for run in params)
    assert all(0.2 <= run.fric_dyn_pp <= 0.8 for run in params)


def test_launch_di_optimal_reads_existing_case_settings(tmp_path, ofat_json):
    old_study = tmp_path / "ofat"
    for index, cor in enumerate((0.1, 0.5, 0.9)):
        case = old_study / f"case_{index}"
        case.mkdir(parents=True)
        (case / "settings.json").write_text(json.dumps({"cor_pp": cor}))

    with patch("rocky_uniaxc.doe.di_optimal.launch_param_cases") as launch:
        launch_di_optimal(
            sweep_name=str(tmp_path / "optimal"),
            scheduler=RockyScheduler.bb_cpu(),
            json_path=ofat_json,
            existing_runs=old_study,
            factors=["cor_pp"],
            bounds=[(0.1, 0.9)],
            n_samples=2,
            n_candidates=32,
            seed=2,
            autolaunch=False,
        )

    assert len(launch.call_args.args[2]) == 2


def test_di_optimal_rejects_out_of_bounds_existing_run(ofat_json):
    with pytest.raises(ValueError, match="within bounds"):
        iter_di_optimal(
            ofat_json,
            [{"cor_pp": 2.0}],
            factors=["cor_pp"],
            bounds=[(0.0, 1.0)],
            n_samples=1,
            n_candidates=8,
        )


def test_d_optimal_selection_accounts_for_existing_runs(ofat_json):
    common = dict(
        json_path=ofat_json,
        factors=["cor_pp"],
        bounds=[(0.1, 0.9)],
        n_samples=1,
        n_candidates=64,
        degree=1,
        seed=4,
    )

    augment_low = iter_di_optimal(existing_runs=[{"cor_pp": 0.1}], **common)
    augment_high = iter_di_optimal(existing_runs=[{"cor_pp": 0.9}], **common)

    assert augment_low[0].cor_pp > 0.8
    assert augment_high[0].cor_pp < 0.2
