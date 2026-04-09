import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import jinja2

from . import _tqdm_launch, shapes_module_path
from ._doe_utils import (
    case_directory,
    prepare_case,
)
from ..compr_meshgen import create_meshes
from ..utils import slurm_sbatch
from .. import BACKEND

logger = logging.getLogger(__name__)


def iter_ofat(json_path: str, ofat_values: dict[str, list | str], n_points: int):
    """
    Iterate over all parameter combinations.

    Args:
        json_path: Path to json config for sweep
        ofat_values: A dictionary containing the OFAT parameters, test range.
            Accepts the following keys: 'parameters', 'test_range', and 'hold_values'.
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
        params["inseractions"]["pp"]["fric_dyn"],
        params["inseractions"]["pp"]["fric_stat"],
        params["inseractions"]["pp"]["fric_rolling"],
        params["inseractions"]["pp"]["cor"],
        params["inseractions"]["pw"]["fric_dyn"],
        params["inseractions"]["pw"]["fric_stat"],
        params["inseractions"]["pw"]["cor"],
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
        "fric_dyn_pp": params["inseractions"]["pp"]["fric_dyn"],
        "fric_stat_pp": params["inseractions"]["pp"]["fric_stat"],
        "fric_rolling_pp": params["inseractions"]["pp"]["fric_rolling"],
        "cor_pp": params["inseractions"]["pp"]["cor"],
        "fric_dyn_pw": params["inseractions"]["pw"]["fric_dyn"],
        "fric_stat_pw": params["inseractions"]["pw"]["fric_stat"],
        "cor_pw": params["inseractions"]["pw"]["cor"],
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
        "fric_dyn_pp": (0, None),
        "fric_stat_pp": (0, None),
        "fric_rolling_pp": (0, None),
        "cor_pp": (0, 1),
        "fric_dyn_pw": (0, None),
        "fric_stat_pw": (0, None),
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
        elif not (lb <= lb_i <= ub and lb <= ub_i <= ub):
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
            if levels_i.size % 2 == 0:
                hold_i = levels_i[levels_i.size // 2 - 1 : levels_i.size // 2 + 1]
            else:
                hold_i = levels_i[levels_i.size // 2]
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
    autolaunch: bool,
    json_path: str,
    ofat_values: dict[str, list | str],
    loc: str = "bb-cpu",
    target: str = "CPU",
    n_points: int = 10,
    ncpus: Optional[int] = None,
    ngpus: int = 1,
    run_days: int = 10,
    template_dir: Optional[str] = None,
    custom_sh: Optional[str] = None,
    backend: Optional[str] = None,
):
    if not backend:
        backend = BACKEND
    if backend not in ["rocky_prepost", "pyrocky"]:
        raise ValueError("backend must be 'rocky_prepost' or 'pyrocky'")

    if template_dir:
        template_dir = Path(template_dir).resolve()
        if not template_dir.exists():
            raise FileNotFoundError(f"Directory {template_dir} does not exist.")

    target = target.upper()
    if target not in ["CPU", "GPU", "MULTI_GPU"]:
        raise ValueError("Select from 'CPU', 'GPU', 'MULTI_GPU'")
    elif target == "MULTI_GPU":
        raise NotImplementedError("Multi GPU use not validated yet")

    if (loc == "bb-cpu" and target == "GPU") or (loc == "az-gpu" and target == "CPU"):
        raise ValueError(f"{target} is not valid for location {loc}")

    target_quoted = f'"{target}"'

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
        json_path=json_path, ofat_values=ofat_values, n_points=n_points
    )

    total_cases = len(experiments_df)
    vars_list = experiments_df.columns.tolist()
    logger.info("Setting up %d OFAT cases...", total_cases)

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

    logger.info("Generating meshes for %d unique sizes...", len(unique_sizes))
    size_to_mesh_dir = {}
    for size in unique_sizes:
        shared_mesh_dir = sweep_path / f"meshes_{size}"
        shared_mesh_dir.mkdir(parents=True, exist_ok=True)
        create_meshes(size, meshsize=0.01, out_dir=str(shared_mesh_dir))
        size_to_mesh_dir[size] = shared_mesh_dir

    logger.info("Generating scripts and preparing jobs...")
    for i, row in experiments_df.iterrows():
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
            "DYNAMIC_FRICTION_PP": exp_dict["fric_dyn_pp"],
            "STATIC_FRICTION_PP": exp_dict["fric_stat_pp"],
            "COR_PP": exp_dict["cor_pp"],
            "DYNAMIC_FRICTION_PW": exp_dict["fric_dyn_pw"],
            "STATIC_FRICTION_PW": exp_dict["fric_stat_pw"],
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

        if exp_dict["rolling"] != "none":
            script_contxt["ROLLING_FRICTION"] = exp_dict["rolling"]
        else:
            script_contxt["ROLLING_FRICTION"] = 0

        prepare_case(case_dir, script_contxt, backend, rocky_template)

        logger.debug("Case %d prepared", i)

        slurm_sbatch(
            str(case_dir),
            loc=loc,
            autolaunch=False,
            custom_msg=custom_sh,
            ncpus=ncpus,
            ngpus=ngpus,
            run_days=run_days,
        )

        logger.info("Case %d/%d prepared", i + 1, total_cases)

    logger.info("\nOFAT experiments:\n%s", experiments_df)

    if autolaunch:
        _tqdm_launch([str(d) for d in case_dirs], total_cases)
