"""Tests for Sobol / Latin Hypercube space-filling DOE generation."""

import pytest
from scipy.stats import qmc

from rocky_uniaxc.doe.space_filling import (
    iter_space_filling,
    sample_space_filling,
)


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


class TestIterSpaceFilling:
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
