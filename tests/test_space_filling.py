"""Tests for Sobol / Latin Hypercube space-filling DOE generation."""

import json
from unittest.mock import patch

import numpy as np
import pytest
from scipy.stats import qmc

from rocky_uniaxc.doe.space_filling import (
    augment_lhs,
    iter_space_filling,
    launch_lhs_augmentation,
    sample_space_filling,
)
from rocky_uniaxc.utils import RockyScheduler


class TestSampleSpaceFilling:
    @pytest.mark.parametrize(
        "sampler",
        [
            qmc.Sobol(d=2, scramble=True, seed=0),
            qmc.LatinHypercube(d=2, seed=0),
        ],
    )
    def test_shape_and_bounds(self, sampler):
        bounds = [(0.1, 0.9), (1e6, 1e7)]
        s = sample_space_filling(bounds, n_samples=8, sampler=sampler)
        assert s.shape == (8, 2)
        for j, (lo, hi) in enumerate(bounds):
            assert s[:, j].min() >= lo
            assert s[:, j].max() <= hi

    def test_reproducible(self):
        a = sample_space_filling(
            [(0, 1)], 8, sampler=qmc.Sobol(d=1, scramble=True, seed=42)
        )
        b = sample_space_filling(
            [(0, 1)], 8, sampler=qmc.Sobol(d=1, scramble=True, seed=42)
        )
        assert (a == b).all()


class TestAugmentLHS:
    def test_fills_each_target_stratum_once(self):
        existing = np.array([[0.05, 0.35], [0.45, 0.65], [0.75, 0.95]])

        new = augment_lhs(existing, [0, 0], [1, 1], 5, seed=42, trials=20)

        combined = np.vstack((existing, new))
        assert new.shape == (2, 2)
        for column in combined.T:
            assert sorted((column * 5).astype(int)) == list(range(5))

    def test_is_reproducible_in_original_bounds(self):
        existing = np.array([[12.0], [18.0]])

        a = augment_lhs(existing, [10], [20], 4, seed=42, trials=20)
        b = augment_lhs(existing, [10], [20], 4, seed=42, trials=20)

        np.testing.assert_array_equal(a, b)
        assert np.all((10 <= a) & (a <= 20))

    def test_collision_handling(self):
        existing = np.array([[0.1], [0.15]])

        with pytest.raises(ValueError, match="stratum collision"):
            augment_lhs(existing, [0], [1], 4, trials=1)

        with pytest.warns(UserWarning, match="closest feasible"):
            new = augment_lhs(
                existing, [0], [1], 4, trials=1, strict=False
            )
        assert new.shape == (2, 1)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"target_size": 2}, "target_size"),
            ({"target_size": 3, "trials": 0}, "trials"),
            ({"target_size": 3, "lower": [0, 0]}, "one bound"),
        ],
    )
    def test_rejects_invalid_inputs(self, kwargs, message):
        arguments = {
            "X": np.array([[0.1], [0.8]]),
            "lower": [0],
            "upper": [1],
            "target_size": 3,
        }
        arguments.update(kwargs)
        with pytest.raises(ValueError, match=message):
            augment_lhs(**arguments)

    def test_rejects_points_outside_original_bounds(self):
        with pytest.raises(ValueError, match="outside"):
            augment_lhs([[2.0]], [0], [1], 2, trials=1)

    def test_launches_only_new_points(self, tmp_path, ofat_json):
        existing = np.array([[0.05], [0.45], [0.75]])

        with patch(
            "rocky_uniaxc.doe.space_filling.launch_param_cases"
        ) as launch:
            launch_lhs_augmentation(
                sweep_name=str(tmp_path / "lhs_extension"),
                scheduler=RockyScheduler.bb_cpu(),
                json_path=ofat_json,
                existing_points=existing,
                factors=["cor_pp"],
                bounds=[(0.0, 1.0)],
                target_size=5,
                seed=42,
                trials=20,
                autolaunch=False,
            )

        new_params = launch.call_args.args[2]
        assert len(new_params) == 2
        combined = np.append(existing[:, 0], [p.cor_pp for p in new_params])
        assert sorted((combined * 5).astype(int)) == list(range(5))
        assert launch.call_args.kwargs["autolaunch"] is False


class TestIterSpaceFilling:
    def test_sphere_uses_default_shape_parameters(self, ofat_json, tmp_path):
        config = json.loads(tmp_path.joinpath(ofat_json).read_text())
        config["shape"] = {"name": "sphere"}
        json_path = tmp_path / "sphere.json"
        json_path.write_text(json.dumps(config))

        params = iter_space_filling(
            str(json_path),
            ["cor_pp"],
            [(0.1, 0.9)],
            n_samples=1,
            sampler=qmc.LatinHypercube(d=1, seed=0),
        )

        assert params[0].shape.name == "sphere"
        assert params[0].shape.vert_ar == 1.0
        assert params[0].shape.horiz_ar == 1.0
        assert params[0].shape.n_corners == 6
        assert params[0].shape.sq_degree == 2.0

    def test_builds_sim_params_in_bounds(self, ofat_json):
        factors = ["cor_pp", "youngmod"]
        bounds = [(0.1, 0.9), (1e6, 1e7)]
        params = iter_space_filling(
            ofat_json,
            factors,
            bounds,
            n_samples=8,
            sampler=qmc.LatinHypercube(d=2, seed=0),
        )
        assert len(params) == 8
        for p in params:
            assert 0.1 <= p.cor_pp <= 0.9
            assert 1e6 <= p.youngmod <= 1e7
            # untouched factor stays at its base value
            assert p.density == 2700

    def test_n_corners_rounded_to_int(self, ofat_json):
        params = iter_space_filling(
            ofat_json,
            ["n_corners"],
            [(10, 50)],
            n_samples=4,
            sampler=qmc.LatinHypercube(d=1, seed=0),
        )
        for p in params:
            assert isinstance(p.shape.n_corners, int)

    def test_rejects_categorical_factor(self, ofat_json):
        with pytest.raises(ValueError, match="categorical"):
            iter_space_filling(
                ofat_json,
                ["normal"],
                [(0, 1)],
                4,
                sampler=qmc.LatinHypercube(d=1, seed=0),
            )

    def test_rejects_unknown_factor(self, ofat_json):
        with pytest.raises(ValueError, match="Unknown"):
            iter_space_filling(
                ofat_json,
                ["not_a_param"],
                [(0, 1)],
                4,
                sampler=qmc.LatinHypercube(d=1, seed=0),
            )

    def test_mismatched_lengths(self, ofat_json):
        with pytest.raises(ValueError, match="same length"):
            iter_space_filling(
                ofat_json,
                ["cor_pp"],
                [(0, 1), (0, 1)],
                4,
                sampler=qmc.LatinHypercube(d=1, seed=0),
            )

    def test_inverted_bounds(self, ofat_json):
        with pytest.raises(ValueError, match="Invalid bounds"):
            iter_space_filling(
                ofat_json,
                ["cor_pp"],
                [(0.9, 0.1)],
                4,
                sampler=qmc.LatinHypercube(d=1, seed=0),
            )
