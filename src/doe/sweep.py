#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sympy.physics.units import Pa

__author__ = "Abhirup Roy"
__email__ = "axr154@bham.ac.uk"
__status__ = "Development"

import os
import json
import subprocess
from collections import OrderedDict
import itertools
from typing import Optional

import jinja2

from .ofat import launch_ofat
from . import _tqdm_launch, shapes_module_path
from ..compr_meshgen import create_meshes_efficiently
from ..utils import slurm_sbatch, cd

"""
This script generates multiple cases for Rocky DEM simulations using a template and a set of parameters.
It creates a directory for each case, populates the template with parameters, and generates the necessary mesh files.
It also creates a slurm sbatch script for each case and can automatically launch the job on a slurm cluster.
It uses the jinja2 library for templating and the gmsh library for mesh generation.

The script uses a JSON file to define the parameters for the simulations.
The parameters include properties of the particles, interactions, and experimental settings.
The script iterates over all combinations of parameters, creating a case directory for each combination.

Example usage:
    from rocky_sweep import make_cases
    make_cases(
        meshdir='meshes',
        json_path='params.json',
        template_dir='templates',
        autolaunch=True
    )
"""


def iter_params(json_path: str):
    """
    Iterate over all parameter combinations.

    Args:
        json_path: Path to json config for sweep
    """
    # Load the JSON file
    with open(json_path, "r") as f_params:
        params = json.load(f_params, object_pairs_hook=OrderedDict)

    # Handle shape parameters - now it's an array of shape objects
    shape_list = params["shape"]
    if not isinstance(shape_list, list):
        shape_list = [shape_list]

    # Extract all possible values for each shape parameter
    shape_names = []
    vert_ars = []
    horiz_ars = []
    n_corners_list = []
    sq_degrees = []
    particle_paths = []

    for shape in shape_list:
        print(shape)
        shape_names.append(shape.get("name", "sphere"))
        vert_ars.append(shape.get("vert_ar", 1.0))
        horiz_ars.append(shape.get("horiz_ar", 1.0))
        n_corners_list.append(shape.get("n_corners", 6))
        sq_degrees.append(shape.get("sq_degree", 1.0))
        particle_paths.append(shape.get("particle_path", ""))

    # Find all combinations of parameters
    param_combinations = itertools.product(
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
        shape_list,
    )
    return param_combinations


def launch_sweep(
    sweep_name: str,
    json_path: str,
    meshdir: str = "meshes",
    template_dir: Optional[str] = None,
    autolaunch=True,
    loc: str = "bb-gpu",
    custom_sh: Optional[str] = None,
    target: str = "GPU",
    ncpus: Optional[int] = None,
):
    """
    Generate and launch sweep cases

    Args:
        sweep_name: A string for the title of the sweep being carried out
        json_path: A path to the json config for the sweep or simulation
        template_dir: A path to the script templates
        autolaunch: Whether to automatically launch scripts
        loc: Specify cluster script to use. Currently only works with
            'az-gpu', 'bb-cpu', 'custom'. N.B. If using custom, specify
            the `custom_sh` arg
        custom_sh: A custom SLURM script to run simulations. Only needed if
            using `loc=custom`
    """
    # Ensure the template directory exists
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

    # Get all parameter combinations
    all_params = list(iter_params(json_path))
    total_cases = len(all_params)
    print(f"Setting up {total_cases} cases...")

    # Create the sweep directory
    os.makedirs(sweep_name, exist_ok=True)

    # Create directories for all cases first (parallel processing preparation)
    case_dirs = []
    for i in range(total_cases):
        case_dir = os.path.join(sweep_name, f"case_{i}")
        os.makedirs(case_dir, exist_ok=True)
        os.makedirs(os.path.join(case_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(case_dir, meshdir), exist_ok=True)
        case_dirs.append(case_dir)

    # Generate meshes only once per unique size
    # This is a major optimization - don't recreate identical meshes
    unique_sizes = set([params[11] for params in all_params])  # Box length parameter
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
        print(params)

        # Prepare script context
        script_contxt = {
            "RADIUS_P": params[0],
            "DENSITY_P": params[1],
            "POISSON_P": params[2],
            "YOUNGMOD_P": params[3],
            "DYNAMIC_FRICTION_PP": params[4],
            "STATIC_FRICTION_PP": params[5],
            "COR_PP": params[7],
            "DYNAMIC_FRICTION_PW": params[8],
            "STATIC_FRICTION_PW": params[9],
            "COR_PW": params[10],
            "L_BOX": params[11],
            "P_COMPRESS": params[12],
            "NORMAL_MODEL": params[13],
            "TANG_MODEL": params[14],
            "ROLLING_MODEL": params[15],
            "ADH_MODEL": params[16],
            "SHAPE": params[-1].get(
                "name", "sphere"
            ),  # Use the name from the shape object
            "VERT_AR": params[-1].get("vert_ar", 0.5),
            "HORIZ_AR": params[-1].get("horiz_ar", 1.0),
            "N_CORNERS": int(params[-1].get("n_corners", 8)),
            "SQ_DEGREE": params[-1].get("sq_degree", 2.0),
            "PARTICLE_PATH": params[-1].get("particle_path", ""),
            "SMOOTHNESS": params[-1].get("smoothness", 0.5),
            "MESH_DIR": str(meshdir),
            "XPU": target,
            "SHAPES_MODULE_PATH": shapes_module_path,
        }

        if params[15] != '"none"':
            script_contxt["ROLLING_FRICTION"] = params[6]

        print(params)

        # Render template and write script
        rendered_content = rocky_template.render(script_contxt)
        script_path = os.path.join(case_dir, "script_uniax.py")

        with open(script_path, "w") as script_file:
            script_file.write(rendered_content)

        # Log case information
        print(f"Case {i}/{total_cases} prepared")

        # Create SLURM script
        slurm_sbatch(
            case_dir, loc=loc, autolaunch=False, custom_msg=custom_sh, ncpus=ncpus
        )  # Don't launch yet

    # Launch all cases at once if requested
    if autolaunch:
        _tqdm_launch(case_dirs, total_cases)


if __name__ == "__main__":
    """Example of a regular sweep"""
    # make_cases(
    #     sweep_name='reg_sweep_example',
    #     json_path='json/swesp_reg.json',
    #     autolaunch=True,
    #     loc='az-gpu',
    #     target='GPU'
    # )

    """Example of an OFAT sweep"""
    # launch_ofat(
    #     'ofat_example',
    #     autolaunch=True,
    #     json_path='json/ofat_base.json',
    #     ofat_values={
    #         'parameters':['n_corners', 'sq_degree'],
    #         'test_range':[(5, 50), (2.0, 10.0)],
    #         'hold_values': ['m', 'm']
    #     },
    #     n_points=5,
    #     loc='az-gpu',
    #     target='GPU',
    #     ncpus=20
    # )
