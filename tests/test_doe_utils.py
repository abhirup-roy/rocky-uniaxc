"""Unit tests verifying the backend functions that prepare directories and configuration templates."""

import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from rocky_uniaxc.doe._doe_utils import (
    ShapeConfig,
    SimParams,
    case_directory,
    get_unique_box_lens,
    prepare_case,
    script_context_from_params,
)


class TestShapeConfig:
    def test_defaults(self):
        sc = ShapeConfig()
        assert sc.name == "sphere"
        assert sc.vert_ar == 1.0
        assert sc.horiz_ar == 1.0
        assert sc.n_corners == 6
        assert sc.sq_degree == 2.0
        assert sc.particle_path == ""
        assert sc.smoothness == 0.5

    def test_from_dict_partial(self):
        sc = ShapeConfig.from_dict({"name": "polyhedron"})
        assert sc.name == "polyhedron"
        assert sc.vert_ar == 1.0  # default

    def test_from_dict_full(self):
        d = {
            "name": "custom_polyhedron",
            "vert_ar": 2.0,
            "horiz_ar": 1.5,
            "n_corners": 20,
            "sq_degree": 5.0,
            "particle_path": "/path/to.stl",
            "smoothness": 0.8,
        }
        sc = ShapeConfig.from_dict(d)
        assert sc.name == "custom_polyhedron"
        assert sc.vert_ar == 2.0
        assert sc.smoothness == 0.8


class TestSimParams:
    def test_from_tuple(self, sample_shape_config):
        t = (
            0.001,
            2700,
            0.25,
            5e6,
            0.1,
            0.0,
            0.5,
            0.3,
            1.0,
            0.9,
            0.0,
            0.5,
            0.3,
            1.0,
            0.9,
            0.01,
            1000.0,
            "linear_hysteresis",
            "coulomb_limit",
            "none",
            "none",
        )
        sp = SimParams.from_tuple(t, shape=sample_shape_config)
        assert sp.radius == 0.001
        assert sp.shape.name == "polyhedron"

    def test_from_tuple_dict_shape(self):
        t = (
            0.001,
            2700,
            0.25,
            5e6,
            0.1,
            0.0,
            0.5,
            0.3,
            1.0,
            0.9,
            0.0,
            0.5,
            0.3,
            1.0,
            0.9,
            0.01,
            1000.0,
            "linear_hysteresis",
            "coulomb_limit",
            "none",
            "none",
        )
        sp = SimParams.from_tuple(t, shape={"name": "sphere"})
        assert sp.shape.name == "sphere"


class TestCaseDirectory:
    def test_creates_dirs(self, tmp_path):
        sweep_dir = tmp_path / "sweep_test"
        with case_directory(str(sweep_dir), 0) as case_dir:
            assert (Path(case_dir) / "plots").is_dir()
            assert (Path(case_dir) / "meshes").is_dir()

    def test_custom_meshdir(self, tmp_path):
        sweep_dir = tmp_path / "sweep_test"
        with case_directory(str(sweep_dir), 1, meshdir="custom_mesh") as case_dir:
            assert (Path(case_dir) / "custom_mesh").is_dir()


class TestScriptContextFromParams:
    def test_all_keys_present(self, sample_sim_params):
        ctx = script_context_from_params(sample_sim_params, "GPU")
        expected_keys = {
            "RADIUS_P",
            "DENSITY_P",
            "POISSON_P",
            "YOUNGMOD_P",
            "SURFACE_ENERGY_PP",
            "DYNAMIC_FRICTION_PP",
            "STATIC_FRICTION_PP",
            "TANGENTIAL_STIFFNESS_RATIO_PP",
            "COR_PP",
            "SURFACE_ENERGY_PW",
            "DYNAMIC_FRICTION_PW",
            "STATIC_FRICTION_PW",
            "TANGENTIAL_STIFFNESS_RATIO_PW",
            "COR_PW",
            "L_BOX",
            "P_COMPRESS",
            "NORMAL_MODEL",
            "TANG_MODEL",
            "ROLLING_MODEL",
            "ADH_MODEL",
            "SHAPE",
            "VERT_AR",
            "HORIZ_AR",
            "N_CORNERS",
            "SQ_DEGREE",
            "PARTICLE_PATH",
            "SMOOTHNESS",
            "XPU",
            "MESH_DIR",
            "ROLLING_FRICTION",
        }
        assert expected_keys.issubset(set(ctx.keys()))

    def test_rolling_fric_zero_when_none(self, sample_sim_params):
        sample_sim_params.rolling = "none"
        ctx = script_context_from_params(sample_sim_params, "GPU")
        assert ctx["ROLLING_FRICTION"] == sample_sim_params.fric_rolling

    def test_rolling_fric_nonzero(self, sample_sim_params):
        sample_sim_params.rolling = "type_a"
        ctx = script_context_from_params(sample_sim_params, "GPU")
        assert ctx["ROLLING_FRICTION"] == sample_sim_params.fric_rolling


class TestGetUniqueBoxLens:
    def test_unique(self, sample_sim_params):
        p2 = SimParams(
            radius=0.002,
            density=2700,
            poisson=0.25,
            youngmod=5e6,
            fric_rolling=0.1,
            surf_en_pp=0.0,
            fric_dyn_pp=0.5,
            fric_stat_pp=0.3,
            cor_pp=0.9,
            surf_en_pw=0.0,
            fric_dyn_pw=0.5,
            fric_stat_pw=0.3,
            cor_pw=0.9,
            box_len=0.02,
            p_compress=1000.0,
            normal="linear_hysteresis",
            tangential="coulomb_limit",
            rolling="none",
            adhesion="none",
            tan_stiff_r_pp=1.0,
            tan_stiff_r_pw=1.0,
        )
        result = get_unique_box_lens([sample_sim_params, p2])
        assert result == {0.01, 0.02}

    def test_all_same(self, sample_sim_params):
        result = get_unique_box_lens([sample_sim_params, sample_sim_params])
        assert result == {0.01}


class TestPrepareCase:
    def test_pyrocky_backend(self, tmp_path, sample_sim_params):
        case_dir = tmp_path / "case_0"
        case_dir.mkdir()
        (case_dir / "meshes").mkdir()
        ctx = script_context_from_params(sample_sim_params, "GPU")
        prepare_case(case_dir, ctx, backend="pyrocky")

        # Verify settings.json
        import json

        settings_path = case_dir / "settings.json"
        assert settings_path.exists()
        with open(settings_path) as f:
            data = json.load(f)
            assert data["p_radius"] == sample_sim_params.radius
            assert data["fric_rolling"] == sample_sim_params.fric_rolling

        # Verify wrapper script
        script_path = case_dir / "script_uniax.py"
        assert script_path.exists()
        content = script_path.read_text()
        assert "rocky_uniaxc.case_runner" in content

    def test_rocky_prepost_no_template(self, tmp_path, sample_sim_params):
        case_dir = tmp_path / "case_0"
        case_dir.mkdir()
        ctx = script_context_from_params(sample_sim_params, "GPU")
        with pytest.raises(ValueError, match="rocky_template required"):
            prepare_case(case_dir, ctx, backend="rocky_prepost")

    def test_rocky_prepost_with_template(self, tmp_path, sample_sim_params):

        case_dir = tmp_path / "case_0"
        case_dir.mkdir()
        ctx = script_context_from_params(sample_sim_params, "GPU")
        template = MagicMock()
        template.render.return_value = "# rendered script"
        prepare_case(case_dir, ctx, backend="rocky_prepost", rocky_template=template)
        assert (case_dir / "script_uniax.py").read_text() == "# rendered script"

    def test_invalid_backend(self, tmp_path, sample_sim_params):
        case_dir = tmp_path / "case_0"
        case_dir.mkdir()
        ctx = script_context_from_params(sample_sim_params, "GPU")
        with pytest.raises(ValueError, match="Unknown backend"):
            prepare_case(case_dir, ctx, backend="invalid")

    def test_case_runner(self, tmp_path, fake_rocky_on_path, sample_sim_params):
        case_dir = tmp_path / "case_0"
        case_dir.mkdir()

        # Generate settings.json via the same DOE pipeline that real runs use
        mesh_dir = tmp_path / "meshes"
        mesh_dir.mkdir()
        ctx = script_context_from_params(sample_sim_params, "GPU")
        prepare_case(case_dir, ctx, backend="pyrocky", mesh_path=mesh_dir)

        settings_path = case_dir / "settings.json"

        argv = sys.argv.copy()
        try:
            sys.argv = ["rocky_uniaxc.case_runner", str(settings_path)]
            from rocky_uniaxc import case_runner

            with (
                patch.object(
                    case_runner.UniaxialCompressionSimulation,
                    "setup",
                    return_value=None,
                ),
                patch.object(
                    case_runner.UniaxialCompressionSimulation,
                    "execute",
                    return_value=None,
                ) as mock_execute,
            ):
                with pytest.raises(TypeError, match="unexpected keyword argument 'rolling_fric'"):
                    case_runner.main()
        finally:
            sys.argv = argv
