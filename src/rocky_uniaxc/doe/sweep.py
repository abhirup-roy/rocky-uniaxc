"""Full-factorial parameter sweep generation and execution.

Reads a JSON configuration specifying parameter ranges, computes all
combinations via the Cartesian product, generates case directories, mesh
files, simulation scripts, and SLURM submission scripts, and optionally
launches the jobs.
"""

import json
import os
import sys
from collections import OrderedDict
import itertools
from pathlib import Path
from typing import Optional

import jinja2
from tqdm import tqdm

from . import shapes_module_path
from ._doe_utils import (
    SimParams,
    ShapeConfig,
    case_directory,
    script_context_from_params,
    get_unique_box_lens,
    prepare_case,
)
from ..compr_meshgen import create_meshes
from ..utils import RockyScheduler


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

    param_combinations = itertools.product(
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
        shape_list,
    )

    return [
        SimParams.from_tuple(combo[:22], shape=combo[22])
        for combo in param_combinations
    ]


def launch_sweep(
    sweep_name: str,
    scheduler: RockyScheduler,
    json_path: str,
    meshdir: str = "meshes",
    template_dir: Optional[str | os.PathLike] = None,
    autolaunch: bool = True,
    target: str = "GPU",
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
        scheduler: :class:`~rocky_uniaxc.schedulers.RockyScheduler` describing
            the SLURM configuration for each case. Defaults to
            ``RockyScheduler.bb_gpu()`` when ``None``.
        meshdir: Name of the mesh subdirectory inside each case. Defaults to
            ``"meshes"``.
        template_dir: Optional path to a directory containing custom Jinja2
            templates. Defaults to the package's built-in templates.
        autolaunch: Whether to automatically submit SLURM jobs after setup.
            Defaults to ``True``.
        target: Compute target — ``"CPU"`` or ``"GPU"``. Defaults to
            ``"GPU"``.
        backend: Simulation backend — ``"rocky_prepost"`` or ``"pyrocky"``.
            Defaults to the package-level :data:`BACKEND` setting.

    Raises:
        ValueError: If an unsupported backend or target is specified.
        FileNotFoundError: If ``template_dir`` does not exist.
    """
    if backend is None:
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

    sweep_path = Path(sweep_name)
    sweep_path.mkdir(exist_ok=True)

    case_dirs = []
    for i in range(total_cases):
        case_dirs.append(sweep_path / f"case_{i}")

    unique_sizes = get_unique_box_lens(all_params)

    size_to_mesh_dir = {}
    for size in tqdm(unique_sizes, desc="Generating meshes", unit="mesh"):
        shared_mesh_dir = sweep_path / f"meshes_{size}"
        shared_mesh_dir.mkdir(parents=True, exist_ok=True)
        create_meshes(size, meshsize=0.01, out_dir=str(shared_mesh_dir))
        size_to_mesh_dir[size] = shared_mesh_dir

    for i, params in tqdm(
        enumerate(all_params),
        total=total_cases,
        desc="Preparing cases",
        unit="case",
    ):
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

        scheduler.generate(case_dir)

    tqdm.write(f"\nAll cases:\n{all_params}")

    if autolaunch:
        scheduler.launch_all([str(d) for d in case_dirs])
