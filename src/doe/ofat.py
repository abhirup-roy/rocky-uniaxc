import os
import json
from collections import OrderedDict
from typing import Optional

import numpy as np
import pandas as pd
import jinja2

from . import _tqdm_launch, shapes_module_path
from ..compr_meshgen import create_meshes_efficiently
from ..utils import slurm_sbatch, cd


def iter_ofat(json_path: str, ofat_values: dict[str, list | str], n_points: int):
    """
    Iterate over all parameter combinations.

    Args:
        json_path: Path to json config for sweep
        ofat_values: A dictionary containing the OFAT parameters, test range.
            Accepts the following keys: 'parameters', 'test_range', and 'hold_values'.
    """
    # Load the JSON file
    with open(json_path, "r") as f_params:
        params = json.load(f_params, object_pairs_hook=OrderedDict)

    # Handle shape parameters - now it's an array of shape objects
    shape = params["shape"]
    if isinstance(shape, list):
        raise ValueError("Shape parameters should be a single object, not a list.")

    # Manual check for parameter types
    # Base case should NOT be a list
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

    # Check if ofat_values is a dictionary and contains the required keys
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
        "radius": params["particle_properties"]["radius"],  # float
        "density": params["particle_properties"]["density"],  # float
        "poisson": params["particle_properties"]["poisson"],  # float
        "youngmod": params["particle_properties"]["youngmod"],  # float
        "fric_dyn_pp": params["inseractions"]["pp"]["fric_dyn"],  # float
        "fric_stat_pp": params["inseractions"]["pp"]["fric_stat"],  # float
        "fric_rolling_pp": params["inseractions"]["pp"]["fric_rolling"],  # float
        "cor_pp": params["inseractions"]["pp"]["cor"],  # float
        "fric_dyn_pw": params["inseractions"]["pw"]["fric_dyn"],  # float
        "fric_stat_pw": params["inseractions"]["pw"]["fric_stat"],  # float
        "cor_pw": params["inseractions"]["pw"]["cor"],  # float
        "box_len": params["experim_settings"]["box_len"],  # float
        "p_compress": params["experim_settings"]["p_compress"],  # float
        "normal": params["contact_model"]["normal"],  # float
        "tangential": params["contact_model"]["tangential"],  # float
        "rolling": params["contact_model"]["rolling"],  # float
        "adhesion": params["contact_model"]["adhesion"],  # float
        "shape": params["shape"]["name"],  # string
        "vert_ar": params["shape"]["vert_ar"],  # float
        "horiz_ar": params["shape"]["horiz_ar"],  # float
        "n_corners": params["shape"]["n_corners"],  # int
        "sq_degree": params["shape"]["sq_degree"],  # float
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
):

    if not template_dir:
        pass
    else:
        template_dir = os.path.abspath(template_dir)
        if not os.path.exists(template_dir):
            raise FileNotFoundError(f"Directory {template_dir} does not exist.")

    target = target.upper()
    if target not in ["CPU", "GPU", "MULTI_GPU"]:
        raise ValueError("Select from 'CPU', 'GPU', 'MULTI_GPU'")
    elif target == "MULTI_GPU":
        raise NotImplementedError("Multi GPU use not validated yet")

    if (loc == "bb-cpu" and target == "GPU") or (loc == "az-gpu" and target == "CPU"):
        raise ValueError(f"{target} is not valid for location {loc}")
    target = '"' + target + '"'
    # Load template once
    if not template_dir:
        rocky_templ_env = jinja2.Environment(
            loader=jinja2.PackageLoader("rocky_uniaxc", "templates"),
        )
    else:
        rocky_templ_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(f"{template_dir}")
        )
    rocky_template = rocky_templ_env.get_template("template_uniax.py")

    experiments_df, base_dict = iter_ofat(
        json_path=json_path, ofat_values=ofat_values, n_points=n_points
    )

    total_cases = len(experiments_df)
    vars = experiments_df.columns.tolist()

    # Generate all cases
    all_params = []

    for _, row in experiments_df.iterrows():
        exp_dict = {}
        for i in range(len(vars)):
            exp_dict[vars[i]] = row[vars[i]]
        exp_dict.update(base_dict)
        all_params.append(exp_dict)

    # Create the sweep directory
    os.makedirs(sweep_name, exist_ok=True)

    # Create directories for all cases first (parallel processing preparation)
    case_dirs = []
    for i in range(total_cases):
        case_dir = os.path.join(sweep_name, f"case_{i}")
        os.makedirs(case_dir, exist_ok=True)
        os.makedirs(os.path.join(case_dir, "plots"), exist_ok=True)
        case_dirs.append(case_dir)

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

    print(f"Generating meshes for {len(unique_sizes)} unique sizes...")
    for size in unique_sizes:
        # Create a shared mesh directory for this size
        shared_mesh_dir = os.path.join(sweep_name, f"meshes_{size}")
        os.makedirs(shared_mesh_dir, exist_ok=True)

        # Generate meshes only once for each unique size
        create_meshes_efficiently(size, meshsize=0.01, out_dir=shared_mesh_dir)

        size_to_mesh_dir[size] = shared_mesh_dir

    # Write scripts and prepare to launch
    print("Generating scripts and preparing jobs...")
    for i, params in enumerate(all_params):
        case_dir = case_dirs[i]

        # Prepare script context
        script_contxt = {
            "RADIUS_P": params["radius"],
            "DENSITY_P": params["density"],
            "POISSON_P": params["poisson"],
            "YOUNGMOD_P": params["youngmod"],
            "DYNAMIC_FRICTION_PP": params["fric_dyn_pp"],
            "STATIC_FRICTION_PP": params["fric_stat_pp"],
            "COR_PP": params["cor_pp"],
            "DYNAMIC_FRICTION_PW": params["fric_dyn_pw"],
            "STATIC_FRICTION_PW": params["fric_stat_pp"],
            "COR_PW": params["cor_pw"],
            "L_BOX": params["box_len"],
            "P_COMPRESS": params["p_compress"],
            "NORMAL_MODEL": params["normal"],
            "TANG_MODEL": params["tangential"],
            "ROLLING_MODEL": params["rolling"],
            "ADH_MODEL": params["adhesion"],
            "SHAPE": params["shape"],
            "VERT_AR": params.get("vert_ar"),
            "HORIZ_AR": params.get("horiz_ar"),
            "N_CORNERS": int(params.get("n_corners")),
            "SQ_DEGREE": params.get("sq_degree"),
            "PARTICLE_PATH": params.get("particle_path"),
            "SMOOTHNESS": params.get("smoothness", 0.5),
            "XPU": target,
            "MESH_DIR": "meshes",
            "SHAPES_MODULE_PATH": shapes_module_path,
        }

        if params["rolling"] != "none":
            script_contxt["ROLLING_FRICTION"] = params["rolling"]
        else:
            script_contxt["ROLLING_FRICTION"] = 0

        rendered_content = rocky_template.render(script_contxt)
        script_path = os.path.join(case_dir, "script_uniax.py")

        with open(script_path, "w") as script_file:
            script_file.write(rendered_content)

        # Log case information
        print(f"Case {i + 1}/{total_cases} prepared")

        # Create SLURM script
        slurm_sbatch(
            case_dir,
            loc=loc,
            autolaunch=False,
            custom_msg=custom_sh,
            ncpus=ncpus,
            ngpus=ngpus,
            run_days=run_days,
        )  # Don't launch yet

    # Launch all cases at once if requested
    print("\nOFAT experiments:\n", experiments_df)
    if autolaunch:
        _tqdm_launch(case_dirs, total_cases)
