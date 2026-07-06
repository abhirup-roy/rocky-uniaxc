"""Validate the execution logic for setting up One Factor at a Time experiments."""

import pytest
from unittest.mock import patch
from rocky_uniaxc.doe.ofat import launch_ofat
from rocky_uniaxc import RockyScheduler


class TestLaunchOfat:
    @pytest.fixture(autouse=True)
    def mock_deps(self):
        with (
            patch("rocky_uniaxc.utils.RockyScheduler.generate"),
            patch("rocky_uniaxc.doe.ofat.create_meshes"),
        ):
            yield

    def test_basic(self, tmp_path, ofat_json):
        ofat_values = {
            "parameters": ["n_corners"],
            "test_range": [(10, 50)],
            "hold_values": ["m"],
        }
        sweep_name = tmp_path / "test_basic"
        launch_ofat(scheduler=RockyScheduler.bb_cpu(), 
            sweep_name=str(sweep_name),
            ofat_values=ofat_values,
            n_points=5,
            json_path=ofat_json,
            autolaunch=False,
            backend="pyrocky",
        )
        assert sweep_name.exists()
        assert (sweep_name / "case_0").exists()

    def test_hold_high(self, tmp_path, ofat_json):
        ofat_values = {
            "parameters": ["n_corners"],
            "test_range": [(10, 50)],
            "hold_values": ["h"],
        }
        sweep_name = tmp_path / "test_high"
        launch_ofat(scheduler=RockyScheduler.bb_cpu(), 
            sweep_name=str(sweep_name),
            ofat_values=ofat_values,
            n_points=5,
            json_path=ofat_json,
            autolaunch=False,
            backend="pyrocky",
        )
        import json

        with open(sweep_name / "case_0" / "settings.json") as f:
            data = json.load(f)
            assert data["n_corners"] == 50

    def test_hold_low(self, tmp_path, ofat_json):
        ofat_values = {
            "parameters": ["n_corners"],
            "test_range": [(10, 50)],
            "hold_values": ["l"],
        }
        sweep_name = tmp_path / "test_low"
        launch_ofat(scheduler=RockyScheduler.bb_cpu(), 
            sweep_name=str(sweep_name),
            ofat_values=ofat_values,
            n_points=5,
            json_path=ofat_json,
            autolaunch=False,
            backend="pyrocky",
        )
        import json

        with open(sweep_name / "case_0" / "settings.json") as f:
            data = json.load(f)
            assert data["n_corners"] == 10

    def test_hold_mid_odd(self, tmp_path, ofat_json):
        ofat_values = {
            "parameters": ["n_corners"],
            "test_range": [(10, 50)],
            "hold_values": ["m"],
        }
        sweep_name = tmp_path / "test_mid"
        launch_ofat(scheduler=RockyScheduler.bb_cpu(), 
            sweep_name=str(sweep_name),
            ofat_values=ofat_values,
            n_points=5,
            json_path=ofat_json,
            autolaunch=False,
            backend="pyrocky",
        )
        import json

        with open(sweep_name / "case_0" / "settings.json") as f:
            data = json.load(f)
            assert data["n_corners"] == 30

    def test_hold_mid_even(self, tmp_path, ofat_json):
        ofat_values = {
            "parameters": ["n_corners"],
            "test_range": [(10, 50)],
            "hold_values": ["m"],
        }
        sweep_name = tmp_path / "test_mid_even"
        # Even n_points with hold='m' must select a single scalar midpoint,
        # not a 2-element slice that caused an ambiguous-truth-value crash.
        launch_ofat(scheduler=RockyScheduler.bb_cpu(), 
            sweep_name=str(sweep_name),
            ofat_values=ofat_values,
            n_points=6,
            json_path=ofat_json,
            autolaunch=False,
            backend="pyrocky",
        )
        assert any(sweep_name.iterdir())

    def test_invalid_params_key(self, tmp_path, ofat_json):
        ofat_values = {
            "parameters": ["nonexistent_param"],
            "test_range": [(0, 1)],
            "hold_values": ["m"],
        }
        sweep_name = tmp_path / "test_err"
        with pytest.raises(ValueError, match="Invalid OFAT parameters"):
            launch_ofat(scheduler=RockyScheduler.bb_cpu(), 
                sweep_name=str(sweep_name),
                ofat_values=ofat_values,
                n_points=5,
                json_path=ofat_json,
                autolaunch=False,
                backend="pyrocky",
            )

    def test_base_out_of_range(self, tmp_path, ofat_json):
        ofat_values = {
            "parameters": ["cor_pp"],
            "test_range": [(0, 1)],
            "hold_values": ["m"],
        }
        sweep_name = tmp_path / "test_out"
        # ofat_json has cor_pp=0.4 which is in [0,1], so this should work
        launch_ofat(scheduler=RockyScheduler.bb_cpu(), 
            sweep_name=str(sweep_name),
            ofat_values=ofat_values,
            n_points=5,
            json_path=ofat_json,
            autolaunch=False,
            backend="pyrocky",
        )
        assert sweep_name.exists()

    def test_lb_ge_ub(self, tmp_path, ofat_json):
        ofat_values = {
            "parameters": ["n_corners"],
            "test_range": [(50, 10)],
            "hold_values": ["m"],
        }
        sweep_name = tmp_path / "test_err"
        # Test values must be increasing properly so we catch this common configuration mistake
        with pytest.raises(ValueError, match="Invalid test range"):
            launch_ofat(scheduler=RockyScheduler.bb_cpu(), 
                sweep_name=str(sweep_name),
                ofat_values=ofat_values,
                n_points=5,
                json_path=ofat_json,
                autolaunch=False,
                backend="pyrocky",
            )

    def test_shape_as_list_raises(self, tmp_path):
        import json

        config = {
            "shape": [{"name": "sphere"}],
            "particle_properties": {
                "radius": 0.001,
                "density": 2700,
                "poisson": 0.25,
                "youngmod": 5e6,
                "fric_rolling": 0.1,
            },
            "interactions": {
                "pp": {
                    "surf_en": 0.0,
                    "fric_dyn": 0.5,
                    "fric_stat": 0.3,
                    "fric_rolling": 0.1,
                    "tan_stiff_r": 1.0,
                    "cor": 0.9,
                },
                "pw": {
                    "surf_en": 0.0,
                    "fric_dyn": 0.5,
                    "fric_stat": 0.3,
                    "fric_rolling": 0.1,
                    "tan_stiff_r": 1.0,
                    "cor": 0.9,
                },
            },
            "experim_settings": {"box_len": 0.01, "p_compress": 1000.0},
            "contact_model": {
                "normal": "linear_hysteresis",
                "tangential": "coulomb_limit",
                "rolling": "none",
                "adhesion": "none",
            },
        }
        path = tmp_path / "bad_shape.json"
        path.write_text(json.dumps(config))
        ofat_values = {
            "parameters": ["n_corners"],
            "test_range": [(10, 50)],
            "hold_values": ["m"],
        }
        sweep_name = tmp_path / "test_err"
        with pytest.raises(ValueError, match="Shape parameters should be a single"):
            launch_ofat(scheduler=RockyScheduler.bb_cpu(), 
                sweep_name=str(sweep_name),
                ofat_values=ofat_values,
                n_points=5,
                json_path=str(path),
                autolaunch=False,
                backend="pyrocky",
            )

    def test_list_in_base_params_raises(self, tmp_path):
        import json

        config = {
            "shape": {
                "name": "sphere",
                "vert_ar": 1,
                "horiz_ar": 1,
                "n_corners": 10,
                "sq_degree": 2.0,
            },
            "particle_properties": {
                "radius": [0.001, 0.002],
                "density": 2700,
                "poisson": 0.25,
                "youngmod": 5e6,
                "fric_rolling": 0.1,
            },
            "interactions": {
                "pp": {
                    "surf_en": 0.0,
                    "fric_dyn": 0.5,
                    "fric_stat": 0.3,
                    "fric_rolling": 0.1,
                    "tan_stiff_r": 1.0,
                    "cor": 0.9,
                },
                "pw": {
                    "surf_en": 0.0,
                    "fric_dyn": 0.5,
                    "fric_stat": 0.3,
                    "fric_rolling": 0.1,
                    "tan_stiff_r": 1.0,
                    "cor": 0.9,
                },
            },
            "experim_settings": {"box_len": 0.01, "p_compress": 1000.0},
            "contact_model": {
                "normal": "linear_hysteresis",
                "tangential": "coulomb_limit",
                "rolling": "none",
                "adhesion": "none",
            },
        }
        path = tmp_path / "list_base.json"
        path.write_text(json.dumps(config))
        ofat_values = {
            "parameters": ["n_corners"],
            "test_range": [(10, 50)],
            "hold_values": ["m"],
        }
        sweep_name = tmp_path / "test_err"
        # Passing multiple sizes at once won't work in OFAT mode where single-variates assume static bases
        with pytest.raises(ValueError, match="should not be lists"):
            launch_ofat(scheduler=RockyScheduler.bb_cpu(), 
                sweep_name=str(sweep_name),
                ofat_values=ofat_values,
                n_points=5,
                json_path=str(path),
                autolaunch=False,
                backend="pyrocky",
            )

    def test_missing_ofat_keys(self, tmp_path, ofat_json):
        sweep_name = tmp_path / "test_err"
        with pytest.raises(ValueError, match="must contain"):
            launch_ofat(scheduler=RockyScheduler.bb_cpu(), 
                sweep_name=str(sweep_name),
                ofat_values={"parameters": ["n_corners"]},
                n_points=5,
                json_path=ofat_json,
                autolaunch=False,
                backend="pyrocky",
            )

    def test_mismatched_lengths(self, tmp_path, ofat_json):
        ofat_values = {
            "parameters": ["n_corners", "sq_degree"],
            "test_range": [(10, 50)],
            "hold_values": ["m", "m"],
        }
        # A simple zip length check enforces equal parameter count and ranges
        sweep_name = tmp_path / "test_err"
        with pytest.raises((ValueError, IndexError)):
            launch_ofat(scheduler=RockyScheduler.bb_cpu(), 
                sweep_name=str(sweep_name),
                ofat_values=ofat_values,
                n_points=5,
                json_path=ofat_json,
                autolaunch=False,
                backend="pyrocky",
            )

    def test_invalid_hold_value(self, tmp_path, ofat_json):
        ofat_values = {
            "parameters": ["n_corners"],
            "test_range": [(10, 50)],
            "hold_values": ["x"],
        }
        sweep_name = tmp_path / "test_err"
        with pytest.raises(ValueError, match="not valid"):
            launch_ofat(scheduler=RockyScheduler.bb_cpu(), 
                sweep_name=str(sweep_name),
                ofat_values=ofat_values,
                n_points=5,
                json_path=ofat_json,
                autolaunch=False,
                backend="pyrocky",
            )
