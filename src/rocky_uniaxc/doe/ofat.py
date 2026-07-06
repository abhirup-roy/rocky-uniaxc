"""One-Factor-at-a-Time (OFAT) experiment setup and execution.

Generates OFAT experiment designs from a JSON base configuration, creates case
directories with simulation scripts and SLURM submission files, and optionally
launches the jobs.
"""

import sys
import os

import json
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import jinja2
from tqdm import tqdm

from . import shapes_module_path
from ._doe_utils import (
    case_directory,
    prepare_case,
)
from ..compr_meshgen import create_meshes
from ..utils import RockyScheduler


def iter_ofat(
    json_path: str, ofat_values: dict[str, list | str], n_points: int
) -> tuple[pd.DataFrame, dict]:
    """Compute all OFAT experiment points from a base configuration.

    Reads the base parameter values from a JSON file and generates an
    experiment matrix where each factor is varied independently while all
    others are held at a designated level.

    Args:
        json_path: Path to the JSON configuration file with base parameters.
        ofat_values: Dictionary specifying the OFAT design. Must contain:

            - ``"parameters"``: list of parameter names to vary.
            - ``"test_range"``: list of ``(lower, upper)`` tuples for each parameter.
            - ``"hold_values"``: list of hold strategies — ``"h"`` (high),
              ``"l"`` (low), or ``"m"`` (mid) — for the baseline of each parameter.

        n_points: Number of evenly-spaced levels to generate for each factor.

    Returns:
        A tuple ``(experiments_df, base_dict)`` where:

        - ``experiments_df`` is a :class:`~pandas.DataFrame` with one row per
            experiment.
        - ``base_dict`` is a dictionary of parameters that remain constant
            across all experiments.

    Raises:
        ValueError: If the OFAT specification is invalid, parameters are
            unrecognised, ranges are out of bounds, or list values appear in
            base parameters.
    """
    with open(json_path, "r") as f_params:
        params = json.load(f_params, object_pairs_hook=OrderedDict)

    shape = params["shape"]
    if isinstance(shape, list):
        raise ValueError("Shape parameters should be a single object, not a list.")

    # Validate no list values in base parameters
    for p in (
        params["particle_properties"]["radius"],
        params["particle_properties"]["density"],
        params["particle_properties"]["poisson"],
        params["particle_properties"]["youngmod"],
        params["interactions"]["pp"]["surf_en"],
        params["interactions"]["pp"]["fric_dyn"],
        params["interactions"]["pp"]["fric_stat"],
        params["interactions"]["pp"]["fric_rolling"],
        params["interactions"]["pp"]["tan_stiff_r"],
        params["interactions"]["pp"]["cor"],
        params["interactions"]["pw"]["surf_en"],
        params["interactions"]["pw"]["fric_dyn"],
        params["interactions"]["pw"]["fric_stat"],
        params["interactions"]["pw"]["fric_rolling"],
        params["interactions"]["pw"]["tan_stiff_r"],
        params["interactions"]["pw"]["cor"],
        params["experim_settings"]["box_len"],
        params["experim_settings"]["p_compress"],
        params["contact_model"]["normal"],
        params["contact_model"]["tangential"],
        params["contact_model"]["rolling"],
        params["contact_model"]["adhesion"],
        shape,
    ):
        if isinstance(p, list):
            raise ValueError(
                f"Parameter values should not be lists. Use a single value for {p}."
            )

    if not isinstance(ofat_values, dict):
        raise ValueError("OFAT values must be provided as a dictionary.")

    ofat_dict_check = set(ofat_values.keys()) == set(
        ["parameters", "test_range", "hold_values"]
    )
    if not ofat_dict_check:
        raise ValueError(
            "OFAT values must contain 'parameters', 'test_range', and 'hold_values' keys."
        )

    ofat_base_valid = {
        "radius": params["particle_properties"]["radius"],
        "density": params["particle_properties"]["density"],
        "poisson": params["particle_properties"]["poisson"],
        "youngmod": params["particle_properties"]["youngmod"],
        "surf_en_pp": params["interactions"]["pp"]["surf_en"],
        "fric_dyn_pp": params["interactions"]["pp"]["fric_dyn"],
        "fric_stat_pp": params["interactions"]["pp"]["fric_stat"],
        "fric_rolling_pp": params["interactions"]["pp"]["fric_rolling"],
        "tan_stiff_r_pp": params["interactions"]["pp"]["tan_stiff_r"],
        "cor_pp": params["interactions"]["pp"]["cor"],
        "surf_en_pw": params["interactions"]["pw"]["surf_en"],
        "fric_dyn_pw": params["interactions"]["pw"]["fric_dyn"],
        "fric_stat_pw": params["interactions"]["pw"]["fric_stat"],
        "fric_rolling_pw": params["interactions"]["pw"]["fric_rolling"],
        "tan_stiff_r_pw": params["interactions"]["pw"]["tan_stiff_r"],
        "cor_pw": params["interactions"]["pw"]["cor"],
        "box_len": params["experim_settings"]["box_len"],
        "p_compress": params["experim_settings"]["p_compress"],
        "normal": params["contact_model"]["normal"],
        "tangential": params["contact_model"]["tangential"],
        "rolling": params["contact_model"]["rolling"],
        "adhesion": params["contact_model"]["adhesion"],
        "shape": params["shape"]["name"],
        "vert_ar": params["shape"]["vert_ar"],
        "horiz_ar": params["shape"]["horiz_ar"],
        "n_corners": params["shape"]["n_corners"],
        "sq_degree": params["shape"]["sq_degree"],
    }

    if not set(ofat_values["parameters"]).issubset(set(ofat_base_valid.keys())):
        raise ValueError(
            f"Invalid OFAT parameters. Allowed parameters are: {list(ofat_base_valid.keys())}"
        )

    range_valid = {
        "surf_en_pp": (0, None),
        "fric_dyn_pp": (0, None),
        "fric_stat_pp": (0, None),
        "fric_rolling_pp": (0, None),
        "tan_stiff_r_pp": (0, None),
        "cor_pp": (0, 1),
        "surf_en_pw": (0, None),
        "fric_dyn_pw": (0, None),
        "fric_stat_pw": (0, None),
        "fric_rolling_pw": (0, None),
        "tan_stiff_r_pw": (0, None),
        "cor_pw": (0, 1),
        "box_len": (0, None),
        "vert_ar": (0, None),
        "horiz_ar": (0, None),
        "n_corners": (10, None),
        "sq_degree": (2, None),
    }

    for k in ofat_values["parameters"]:
        lb, ub = range_valid[k]
        ub = ub if ub is not None else float("inf")
        if k not in ofat_base_valid:
            raise ValueError(f"Parameter '{k}' is not in the base parameters.")
        if not (lb <= ofat_base_valid[k] <= ub):
            raise ValueError(
                f"Base parameter '{k}' with value {ofat_base_valid[k]} is out of range ({lb}, {ub})."
            )

        param_idx = ofat_values["parameters"].index(k)
        test_range = ofat_values["test_range"][param_idx]
        hold_value = ofat_values["hold_values"][param_idx]

        if hold_value not in ["h", "l", "m"]:
            raise ValueError(
                f"Hold value '{hold_value}' for parameter '{k}' is not valid. Use 'h', 'l', or 'm'."
            )

        lb_i, ub_i = test_range
        if lb_i >= ub_i:
            raise ValueError(
                f"Invalid test range for parameter '{k}': ({lb_i}, {ub_i})"
            )
        elif not (lb <= lb_i <= ub and lb <= ub_i <= ub):  # type: ignore
            raise ValueError(
                f"Test range for parameter '{k}' with values ({lb_i}, {ub_i}) is out of bounds ({lb}, {ub})."
            )

    if len(ofat_values["parameters"]) != len(ofat_values["test_range"]) or len(
        ofat_values["hold_values"]
    ) != len(ofat_values["parameters"]):
        raise ValueError("Mismatched lengths in OFAT values.")

    levels = {}
    for i, rng in enumerate(ofat_values["test_range"]):
        lb, ub = rng
        if lb >= ub:
            raise ValueError(
                f"Invalid range for parameter '{ofat_values['parameters'][i]}': ({lb}, {ub})"
            )

        dtype = int if ofat_values["parameters"][i] == "n_corners" else float
        levels_i = np.linspace(lb, ub, n_points, dtype=dtype)
        if ofat_values["hold_values"][i] == "h":
            hold_i = levels_i[-1]
        elif ofat_values["hold_values"][i] == "l":
            hold_i = levels_i[0]
        elif ofat_values["hold_values"][i] == "m":
            hold_i = levels_i[(levels_i.size - 1) // 2]
        else:
            raise ValueError(
                f"Invalid hold value for parameter '{ofat_values['parameters'][i]}':\
                            {ofat_values['hold_values'][i]}. Select from 'h', 'l', 'm'."
            )
        levels[ofat_values["parameters"][i]] = {"levels": levels_i, "hold": hold_i}

    baseline = {param: v["hold"] for param, v in levels.items()}
    experiments = [baseline.copy()]

    for factor, v in levels.items():
        for level in v["levels"]:
            if level != baseline[factor]:
                experiment = baseline.copy()
                experiment[factor] = level
                experiments.append(experiment)

    experiments_df = pd.DataFrame(experiments)

    ofat_vars = set(experiments_df.columns)
    base_vars = set(ofat_base_valid.keys())

    keys_to_drop = base_vars & ofat_vars
    for k in keys_to_drop:
        del ofat_base_valid[k]

    return experiments_df, ofat_base_valid


def launch_ofat(
    sweep_name: str,
    scheduler: RockyScheduler,
    ofat_values: dict[str, list | str],
    n_points: int,
    json_path: str | os.PathLike,
    autolaunch: bool = True,
    target: str = "CPU",
    backend: Optional[str] = None,
    template_dir: Optional[str | os.PathLike] = None,
) -> None:
    """Launch a One-Factor-at-a-Time (OFAT) experiment block.

    Generates the necessary case directories, input scripts, and SLURM
    submission scripts for a series of OFAT experiments based on the provided
    configuration.

    Example::

        ofat_values = {
            "parameters": ["cor_pp", "fric_dyn_pp"],
            "test_range": [(0.1, 0.5), (0.2, 0.8)],
            "hold_values": ["m", "l"],
        }
        launch_ofat(
            sweep_name="ofat_sweep",
            ofat_values=ofat_values,
            n_points=5,
            json_path="config.json",
        )

    Args:
        sweep_name: Name of the OFAT experiment block, used for directory
            naming.
        ofat_values: Dictionary specifying the OFAT design. Must contain
            ``"parameters"`` (list of names), ``"test_range"`` (list of
            ``(min, max)`` tuples), and ``"hold_values"`` (list of ``"h"``,
            ``"l"``, or ``"m"`` strategies).
        n_points: Number of test points to generate for each factor.
        json_path: Path to the JSON configuration file with base parameters.
        scheduler: :class:`~rocky_uniaxc.schedulers.RockyScheduler` describing
            the SLURM configuration for each case. Defaults to
            ``RockyScheduler.bb_cpu()`` when ``None``.
        autolaunch: If ``True``, automatically submit the SLURM jobs after
            setup. Defaults to ``True``.
        target: Compute target — ``"CPU"`` or ``"GPU"``. Defaults to
            ``"CPU"``.
        backend: Simulation backend — ``"rocky_prepost"`` or ``"pyrocky"``.
            Defaults to the package-level :data:`BACKEND` setting.
        template_dir: Optional path to a directory with custom Jinja2
            templates. Must contain ``template_uniax.py``.

    Raises:
        ValueError: If an unsupported backend or target is specified.
        FileNotFoundError: If ``template_dir`` does not exist.
        NotImplementedError: If ``target="MULTI_GPU"`` is requested.
    """

    if not backend:
        from .. import BACKEND

        backend = BACKEND

    if backend not in ["rocky_prepost", "pyrocky"]:
        raise ValueError("backend must be 'rocky_prepost' or 'pyrocky'")
    elif backend == "pyrocky":
        scheduler.run_command = (
            f"{sys.executable} -m rocky_uniaxc.case_runner settings.json"
        )

    if template_dir:
        template_dir = Path(template_dir).resolve()
        if not template_dir.exists():
            raise FileNotFoundError(f"Directory {template_dir} does not exist.")

    target = target.upper()
    if target not in ["CPU", "GPU", "MULTI_GPU"]:
        raise ValueError("Select from 'CPU', 'GPU', 'MULTI_GPU'")
    elif target == "MULTI_GPU":
        raise NotImplementedError("Multi GPU use not validated yet")

    target_quoted = f'"{target}"'
    # =========

    # Load template
    if not template_dir:
        rocky_templ_env = jinja2.Environment(
            loader=jinja2.PackageLoader("rocky_uniaxc", "templates"),
        )
    else:
        rocky_templ_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir))
        )
    rocky_template = rocky_templ_env.get_template("template_uniax.py")

    experiments_df, base_dict = iter_ofat(
        json_path=str(json_path), ofat_values=ofat_values, n_points=n_points
    )

    total_cases = len(experiments_df)
    vars_list = experiments_df.columns.tolist()

    sweep_path = Path(sweep_name)
    sweep_path.mkdir(exist_ok=True)

    case_dirs = []
    for i in range(total_cases):
        case_dirs.append(sweep_path / f"case_{i}")

    if "box_len" in experiments_df.columns:
        unique_sizes = set(experiments_df["box_len"])
    elif "box_len" in base_dict.keys():
        unique_sizes = [base_dict["box_len"]]
    else:
        raise ValueError(
            "No box length parameter found in experiments or base dictionary. "
            "Debugging required"
        )

    size_to_mesh_dir = {}
    for size in tqdm(unique_sizes, desc="Generating meshes", unit="mesh"):
        shared_mesh_dir = sweep_path / f"meshes_{size}"
        shared_mesh_dir.mkdir(parents=True, exist_ok=True)
        create_meshes(size, meshsize=0.01, out_dir=str(shared_mesh_dir))
        size_to_mesh_dir[size] = shared_mesh_dir

    for i, row in tqdm(
        experiments_df.iterrows(),
        total=total_cases,
        desc="Preparing OFAT cases",
        unit="case",
    ):
        case_dir = case_dirs[i]

        with case_directory(sweep_path, i, "meshes"):
            pass

        exp_dict = {var: row[var] for var in vars_list}
        exp_dict.update(base_dict)

        script_contxt = {
            "RADIUS_P": exp_dict["radius"],
            "DENSITY_P": exp_dict["density"],
            "POISSON_P": exp_dict["poisson"],
            "YOUNGMOD_P": exp_dict["youngmod"],
            "SURFACE_ENERGY_PP": exp_dict["surf_en_pp"],
            "DYNAMIC_FRICTION_PP": exp_dict["fric_dyn_pp"],
            "STATIC_FRICTION_PP": exp_dict["fric_stat_pp"],
            "TANGENTIAL_STIFFNESS_RATIO_PP": exp_dict["tan_stiff_r_pp"],
            "COR_PP": exp_dict["cor_pp"],
            "SURFACE_ENERGY_PW": exp_dict["surf_en_pw"],
            "DYNAMIC_FRICTION_PW": exp_dict["fric_dyn_pw"],
            "STATIC_FRICTION_PW": exp_dict["fric_stat_pw"],
            "TANGENTIAL_STIFFNESS_RATIO_PW": exp_dict["tan_stiff_r_pw"],
            "COR_PW": exp_dict["cor_pw"],
            "L_BOX": exp_dict["box_len"],
            "P_COMPRESS": exp_dict["p_compress"],
            "NORMAL_MODEL": exp_dict["normal"],
            "TANG_MODEL": exp_dict["tangential"],
            "ROLLING_MODEL": exp_dict["rolling"],
            "ADH_MODEL": exp_dict["adhesion"],
            "SHAPE": exp_dict["shape"],
            "VERT_AR": exp_dict.get("vert_ar"),
            "HORIZ_AR": exp_dict.get("horiz_ar"),
            "N_CORNERS": int(exp_dict.get("n_corners", 8)),
            "SQ_DEGREE": exp_dict.get("sq_degree"),
            "PARTICLE_PATH": exp_dict.get("particle_path"),
            "SMOOTHNESS": exp_dict.get("smoothness", 0.5),
            "XPU": target_quoted,
            "MESH_DIR": "meshes",
            "SHAPES_MODULE_PATH": shapes_module_path,
        }

        if exp_dict["fric_rolling_pp"] != "none":
            script_contxt["ROLLING_FRICTION_PP"] = exp_dict["fric_rolling_pp"]
        else:
            script_contxt["ROLLING_FRICTION_PP"] = 0

        if exp_dict["fric_rolling_pw"] != "none":
            script_contxt["ROLLING_FRICTION_PW"] = exp_dict["fric_rolling_pw"]
        else:
            script_contxt["ROLLING_FRICTION_PW"] = 0

        prepare_case(
            case_dir,
            script_contxt,
            backend,
            rocky_template,
            mesh_path=size_to_mesh_dir[exp_dict["box_len"]],
        )

        scheduler.generate(case_dir)

    tqdm.write(f"\nOFAT experiments:\n{experiments_df}")

    if autolaunch:
        scheduler.launch_all([str(d) for d in case_dirs])
