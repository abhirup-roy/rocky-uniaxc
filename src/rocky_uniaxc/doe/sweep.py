"""Full-factorial parameter sweep generation and execution.

Reads a JSON configuration specifying parameter ranges, computes all
combinations via the Cartesian product, generates case directories, mesh
files, simulation scripts, and SLURM submission scripts, and optionally
launches the jobs.
"""

import json
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

logger = logging.getLogger(__name__)


def iter_params(json_path: str) -> list[SimParams]:
    """Read a sweep JSON configuration and expand all parameter combinations.

    Args:
        json_path: Path to the JSON configuration file defining parameter
            ranges for the sweep.

    Returns:
        List of :class:`~rocky_uniaxc.doe._doe_utils.SimParams` instances,
        one per parameter combination.
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
    template_dir: Optional[str | os.PathLike] = None,
    autolaunch=True,
    loc: str = "bb-gpu",
    custom_sh: Optional[str] = None,
    target: str = "GPU",
    ncpus: Optional[int] = None,
    backend: Optional[str] = None,
):
    """Generate and launch a full-factorial parameter sweep.

    Reads parameter ranges from a JSON configuration, computes all
    combinations, creates case directories with simulation scripts and SLURM
    submission files, and optionally submits the jobs.

    Args:
        sweep_name: Title of the sweep, used as the root directory name.
        json_path: Path to the JSON configuration file defining parameter
            ranges.
        meshdir: Name of the mesh subdirectory inside each case. Defaults to
            ``"meshes"``.
        template_dir: Optional path to a directory containing custom Jinja2
            templates. Defaults to the package's built-in templates.
        autolaunch: Whether to automatically submit SLURM jobs after setup.
            Defaults to ``True``.
        loc: Cluster location for SLURM scripts. Accepted values are
            ``"bb-gpu"``, ``"bb-cpu"``, ``"az-gpu"``, and ``"custom"``.
        custom_sh: Custom SLURM script content. Only used when
            ``loc="custom"``.
        target: Compute target — ``"CPU"`` or ``"GPU"``. Defaults to
            ``"GPU"``.
        ncpus: Number of CPUs to request (CPU target only).
        backend: Simulation backend — ``"rocky_prepost"`` or ``"pyrocky"``.
            Defaults to the package-level :data:`BACKEND` setting.

    Raises:
        ValueError: If an unsupported backend, target, or location is
            specified.
        FileNotFoundError: If ``template_dir`` does not exist.
    """
    if backend is None:
        from .. import BACKEND
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
        prepare_case(
            case_dir,
            script_contxt,
            backend,
            rocky_template,
            mesh_path=size_to_mesh_dir[params.box_len],
        )

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
