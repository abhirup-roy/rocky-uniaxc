#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
from collections import OrderedDict
import itertools
from pathlib import Path
from typing import Optional

import jinja2

from . import _tqdm_launch, shapes_module_path
from ._doe_utils import (
    SimParams,
    ShapeConfig,
    case_directory,
    script_context_from_params,
    get_unique_box_lens,
    prepare_case,
)
from ..compr_meshgen import create_meshes
from ..utils import slurm_sbatch
from .. import BACKEND

logger = logging.getLogger(__name__)

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


def iter_params(json_path: str) -> list[SimParams]:
    """
    Iterate over all parameter combinations.

    Args:
        json_path: Path to json config for sweep

    Returns:
        List of SimParams instances for each parameter combination
    """
    with open(json_path, "r") as f_params:
        params = json.load(f_params, object_pairs_hook=OrderedDict)

    shape_list = params["shape"]
    if not isinstance(shape_list, list):
        shape_list = [shape_list]

    shape_configs = [ShapeConfig.from_dict(s) for s in shape_list]
    logger.debug("Loaded %d shape configurations", len(shape_configs))

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

    return [
        SimParams.from_tuple(combo[:17], shape=combo[17])
        for combo in param_combinations
    ]


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
    backend: Optional[str] = None,
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
        backend: Which backend to use. Options are 'rocky_prepost' or 'pyrocky'
    """
    if backend is None:
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

    all_params = list(iter_params(json_path))
    total_cases = len(all_params)
    logger.info("Setting up %d cases...", total_cases)

    sweep_path = Path(sweep_name)
    sweep_path.mkdir(exist_ok=True)

    case_dirs = []
    for i in range(total_cases):
        case_dirs.append(sweep_path / f"case_{i}")

    unique_sizes = get_unique_box_lens(all_params)
    logger.info("Generating meshes for %d unique sizes...", len(unique_sizes))

    size_to_mesh_dir = {}
    for size in unique_sizes:
        shared_mesh_dir = sweep_path / f"meshes_{size}"
        shared_mesh_dir.mkdir(parents=True, exist_ok=True)
        create_meshes(size, meshsize=0.01, out_dir=str(shared_mesh_dir))
        size_to_mesh_dir[size] = shared_mesh_dir

    logger.info("Generating scripts and preparing jobs...")
    for i, params in enumerate(all_params):
        case_dir = case_dirs[i]

        with case_directory(sweep_path, i, meshdir):
            pass

        script_contxt = script_context_from_params(params, target_quoted, meshdir)
        script_contxt["SHAPES_MODULE_PATH"] = shapes_module_path
        prepare_case(case_dir, script_contxt, backend, rocky_template)

        logger.debug("Case %d prepared: %s", i, params.shape.name)

        slurm_sbatch(
            str(case_dir),
            loc=loc,
            autolaunch=False,
            custom_msg=custom_sh,
            ncpus=ncpus,
        )

        logger.info("Case %d/%d prepared", i + 1, total_cases)

    logger.info("\nAll cases:\n%s", all_params)

    if autolaunch:
        _tqdm_launch([str(d) for d in case_dirs], total_cases)


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
