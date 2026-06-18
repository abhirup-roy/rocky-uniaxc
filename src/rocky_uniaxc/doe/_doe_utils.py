"""Shared utilities for DOE (Design of Experiments) modules.

Provides common dataclasses and helper functions for parameter handling,
script generation, and case directory management used by both
:mod:`rocky_uniaxc.doe.sweep` and :mod:`rocky_uniaxc.doe.ofat`.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import jinja2

logger = logging.getLogger(__name__)


@dataclass
class ShapeConfig:
    """Configuration for a particle shape in simulations.

    Attributes:
        name: Shape identifier (e.g. ``"sphere"``, ``"polyhedron"``).
        vert_ar: Vertical aspect ratio.
        horiz_ar: Horizontal aspect ratio.
        n_corners: Number of corners for polyhedral shapes.
        sq_degree: Superquadric degree.
        particle_path: File path to an STL for custom polyhedra.
        smoothness: Surface smoothness parameter.
    """

    name: str = "sphere"
    vert_ar: float = 1.0
    horiz_ar: float = 1.0
    n_corners: int = 6
    sq_degree: float = 2.0
    particle_path: str = ""
    smoothness: float = 0.5

    @classmethod
    def from_dict(cls, d: dict) -> ShapeConfig:
        """Create a ShapeConfig from a dictionary.

        Missing keys are filled with class defaults.

        Args:
            d: Dictionary of shape configuration values.

        Returns:
            A new ``ShapeConfig`` instance.
        """
        return cls(
            name=d.get("name", "sphere"),
            vert_ar=d.get("vert_ar", 1.0),
            horiz_ar=d.get("horiz_ar", 1.0),
            n_corners=d.get("n_corners", 6),
            sq_degree=d.get("sq_degree", 2.0),
            particle_path=d.get("particle_path", ""),
            smoothness=d.get("smoothness", 0.5),
        )


@dataclass
class SimParams:
    """Typed representation of simulation parameters.

    Replaces magic-index tuple access with named fields for clarity.

    Attributes:
        radius: Particle radius in metres.
        density: Particle density in kg/m³.
        poisson: Poisson's ratio.
        youngmod: Young's modulus in Pa.
        fric_dyn_pp: Dynamic friction coefficient (particle–particle).
        fric_stat_pp: Static friction coefficient (particle–particle).
        fric_rolling_pp: Rolling friction coefficient (particle–particle).
        cor_pp: Coefficient of restitution (particle–particle).
        fric_dyn_pw: Dynamic friction coefficient (particle–wall).
        fric_stat_pw: Static friction coefficient (particle–wall).
        cor_pw: Coefficient of restitution (particle–wall).
        box_len: Length of the simulation box in metres.
        p_compress: Compression pressure in Pa.
        normal: Normal contact force model name.
        tangential: Tangential contact force model name.
        rolling: Rolling resistance model name.
        adhesion: Adhesion model name.
        shape: Particle shape configuration.
    """

    radius: float
    density: float
    poisson: float
    youngmod: float
    fric_dyn_pp: float
    fric_stat_pp: float
    fric_rolling_pp: float
    cor_pp: float
    fric_dyn_pw: float
    fric_stat_pw: float
    cor_pw: float
    box_len: float
    p_compress: float
    normal: str
    tangential: str
    rolling: str
    adhesion: str
    shape: ShapeConfig = field(default_factory=ShapeConfig)

    @classmethod
    def from_tuple(cls, params: tuple, shape: ShapeConfig | dict) -> SimParams:
        """Create a SimParams from a parameter tuple and a shape specification.

        Args:
            params: A 17-element tuple of simulation parameters in the
                canonical order (radius, density, …, adhesion).
            shape: A :class:`ShapeConfig` instance or a dictionary that can
                be passed to :meth:`ShapeConfig.from_dict`.

        Returns:
            A new ``SimParams`` instance.
        """
        if isinstance(shape, dict):
            shape = ShapeConfig.from_dict(shape)
        return cls(
            radius=params[0],
            density=params[1],
            poisson=params[2],
            youngmod=params[3],
            fric_dyn_pp=params[4],
            fric_stat_pp=params[5],
            fric_rolling_pp=params[6],
            cor_pp=params[7],
            fric_dyn_pw=params[8],
            fric_stat_pw=params[9],
            cor_pw=params[10],
            box_len=params[11],
            p_compress=params[12],
            normal=params[13],
            tangential=params[14],
            rolling=params[15],
            adhesion=params[16],
            shape=shape,
        )


@contextmanager
def case_directory(sweep_name: str | Path, case_idx: int, meshdir: str = "meshes"):
    """Context manager for creating and managing a case directory.

    Creates the following directory structure::

        sweep_name/case_<case_idx>/
            plots/
            <meshdir>/

    Args:
        sweep_name: Name or path of the sweep directory.
        case_idx: Index of the case.
        meshdir: Name of the mesh subdirectory.

    Yields:
        pathlib.Path: Path to the created case directory.
    """
    sweep_path = Path(sweep_name)
    case_path = sweep_path / f"case_{case_idx}"

    (case_path / "plots").mkdir(parents=True, exist_ok=True)
    (case_path / meshdir).mkdir(parents=True, exist_ok=True)

    yield case_path


def render_pyrocky_script(
    case_dir: str | Path,
    script_contxt: dict,
    meshdir: str = "meshes",
    mesh_path: Optional[str | Path] = None,
) -> None:
    """Render a pyrocky uniaxial compression simulation case.

    Dumps simulation settings to a ``settings.json`` file and creates a small
    launcher script (``script_uniax.py``) that invokes the case runner.

    Args:
        case_dir: Path to the case directory.
        script_contxt: Dictionary containing script template variables.
        meshdir: Name of the mesh subdirectory (used only when
            ``mesh_path`` is ``None``).
        mesh_path: Absolute path to the mesh directory. When provided this
            takes precedence over ``meshdir``, allowing callers to point
            cases at a shared pre-generated mesh directory.
    """
    case_dir = Path(case_dir)
    if mesh_path is None:
        mesh_path = os.path.abspath(case_dir / meshdir)
    else:
        mesh_path = str(os.path.abspath(mesh_path))

    settings_dict = {
        "particle_box_len": script_contxt["L_BOX"],
        "t_fill": 1.0,
        "t_settle": 0.5,
        "t_compress": 2.0,
        "p_compress": script_contxt["P_COMPRESS"],
        "p_radius": script_contxt["RADIUS_P"],
        "p_density": script_contxt["DENSITY_P"],
        "p_youngmod": script_contxt["YOUNGMOD_P"],
        "p_poisson": script_contxt["POISSON_P"],
        "fric_dyn_pp": script_contxt["DYNAMIC_FRICTION_PP"],
        "fric_stat_pp": script_contxt["STATIC_FRICTION_PP"],
        "cor_pp": script_contxt["COR_PP"],
        "fric_dyn_pw": script_contxt["DYNAMIC_FRICTION_PW"],
        "fric_stat_pw": script_contxt["STATIC_FRICTION_PW"],
        "cor_pw": script_contxt["COR_PW"],
        "normal_force_model": script_contxt["NORMAL_MODEL"].strip('"'),
        "tangential_force_model": script_contxt["TANG_MODEL"].strip('"'),
        "adhesion_model": script_contxt["ADH_MODEL"].strip('"'),
        "rolling_fric": script_contxt.get("ROLLING_FRICTION", 0.0),
        "rolling_model": script_contxt["ROLLING_MODEL"].strip('"'),
        "processor": script_contxt["XPU"].strip('"'),
        "mesh_dir": mesh_path,
        "shape_name": script_contxt["SHAPE"].strip('"'),
        "vert_ar": script_contxt["VERT_AR"],
        "horiz_ar": script_contxt["HORIZ_AR"],
        "n_corners": script_contxt["N_CORNERS"],
        "sq_degree": script_contxt["SQ_DEGREE"],
        "particle_path": script_contxt["PARTICLE_PATH"],
        "smoothness": script_contxt["SMOOTHNESS"],
    }

    settings_path = case_dir / "settings.json"
    with open(settings_path, "w") as f:
        json.dump(settings_dict, f, indent=4)

    script_content = f"""import sys
import subprocess
from pathlib import Path

# Run the single runner module
subprocess.run([sys.executable, "-m", "rocky_uniaxc.case_runner", "settings.json"], check=True)
"""
    (case_dir / "script_uniax.py").write_text(script_content)


def script_context_from_params(
    params: SimParams, target: str, meshdir: str = "meshes"
) -> dict:
    """Build a script context dictionary from a :class:`SimParams` instance.

    Args:
        params: Simulation parameters.
        target: Processor target (``"CPU"``, ``"GPU"``, etc.).
        meshdir: Name of the mesh subdirectory.

    Returns:
        Dictionary of template variables for script rendering.
    """
    rolling_fric = params.fric_rolling_pp if params.rolling != "none" else 0

    return {
        "RADIUS_P": params.radius,
        "DENSITY_P": params.density,
        "POISSON_P": params.poisson,
        "YOUNGMOD_P": params.youngmod,
        "DYNAMIC_FRICTION_PP": params.fric_dyn_pp,
        "STATIC_FRICTION_PP": params.fric_stat_pp,
        "COR_PP": params.cor_pp,
        "DYNAMIC_FRICTION_PW": params.fric_dyn_pw,
        "STATIC_FRICTION_PW": params.fric_stat_pw,
        "COR_PW": params.cor_pw,
        "L_BOX": params.box_len,
        "P_COMPRESS": params.p_compress,
        "NORMAL_MODEL": params.normal,
        "TANG_MODEL": params.tangential,
        "ROLLING_MODEL": params.rolling,
        "ADH_MODEL": params.adhesion,
        "SHAPE": params.shape.name,
        "VERT_AR": params.shape.vert_ar,
        "HORIZ_AR": params.shape.horiz_ar,
        "N_CORNERS": int(params.shape.n_corners),
        "SQ_DEGREE": params.shape.sq_degree,
        "PARTICLE_PATH": params.shape.particle_path,
        "SMOOTHNESS": params.shape.smoothness,
        "XPU": target,
        "MESH_DIR": meshdir,
        "ROLLING_FRICTION": rolling_fric,
    }


def get_unique_box_lens(params_list: list[SimParams]) -> set[float]:
    """Get unique box lengths from a list of :class:`SimParams`.

    Args:
        params_list: List of simulation parameter instances.

    Returns:
        Set of unique ``box_len`` values.
    """
    return {p.box_len for p in params_list}


def prepare_case(
    case_dir: Path,
    script_contxt: dict,
    backend: str,
    rocky_template: Optional[jinja2.Template] = None,
    mesh_path: Optional[str | Path] = None,
) -> None:
    """Write a simulation script to the case directory.

    Args:
        case_dir: Path to the case directory.
        script_contxt: Script context dictionary for template rendering.
        backend: Simulation backend — ``"rocky_prepost"`` or ``"pyrocky"``.
        rocky_template: Jinja2 template instance. Required when
            ``backend="rocky_prepost"``.
        mesh_path: Absolute path to the mesh directory (pyrocky backend
            only). When provided, the generated ``settings.json`` points
            at this directory instead of deriving a per-case path.

    Raises:
        ValueError: If ``backend="rocky_prepost"`` and no template is
            provided, or if the backend string is unrecognised.
    """
    script_path = case_dir / "script_uniax.py"

    if backend == "rocky_prepost":
        if rocky_template is None:
            raise ValueError("rocky_template required for rocky_prepost backend")
        rendered = rocky_template.render(script_contxt)
        script_path.write_text(rendered)
    elif backend == "pyrocky":
        render_pyrocky_script(case_dir, script_contxt, mesh_path=mesh_path)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    logger.debug("Script written to %s", script_path)
