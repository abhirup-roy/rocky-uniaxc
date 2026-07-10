"""Space-filling DOE designs (Sobol sequences / Latin Hypercube sampling).

Unlike the full-factorial sweep, which takes the Cartesian product of discrete
levels, a space-filling design draws ``n_samples`` points that cover a
continuous parameter box as uniformly as possible. This scales to many factors
without the combinatorial blow-up of a grid.

Continuous factors and their ``(min, max)`` bounds are supplied by the caller;
every other parameter is held at the single value read from the JSON base
configuration (same schema as the OFAT config, see :mod:`rocky_uniaxc.doe.ofat`).
"""

import json
from collections import OrderedDict
from typing import Optional

import numpy as np
from scipy.stats import qmc

from ._doe_utils import ShapeConfig, SimParams, resolve_base_params
from .sweep import launch_param_cases

# Parameters that name a model/shape choice — not sample-able on a continuous scale.
_CATEGORICAL = {"normal", "tangential", "rolling", "adhesion", "shape"}
# Sampled factors that belong to the particle shape rather than SimParams proper.
_SHAPE_FACTORS = {"vert_ar", "horiz_ar", "n_corners", "sq_degree"}


def sample_space_filling(
    bounds: list[tuple[float, float]],
    n_samples: int,
    sampler: qmc.QMCEngine,
) -> np.ndarray:
    """Draw a space-filling sample and scale it to the given bounds.

    Args:
        bounds: ``(lower, upper)`` pair for each dimension.
        n_samples: Number of points to draw. For ``method="sobol"`` a power of
            two is recommended; other counts still work but lose the
            sequence's balance property (scipy emits a warning).
        sampler: The QMCEngine instance to use for sampling.

    Returns:
        Array of shape ``(n_samples, len(bounds))`` with values inside the
        supplied bounds.

    """
    unit_sample = sampler.random(n_samples)
    l_bounds = [b[0] for b in bounds]
    u_bounds = [b[1] for b in bounds]
    return qmc.scale(unit_sample, l_bounds, u_bounds)


def _build_sim_params(flat: dict) -> SimParams:
    """Assemble a :class:`SimParams` from a flat parameter dict."""
    shape = ShapeConfig(
        name=flat["shape"],
        vert_ar=flat["vert_ar"],
        horiz_ar=flat["horiz_ar"],
        n_corners=int(round(flat["n_corners"])),
        sq_degree=flat["sq_degree"],
    )
    return SimParams(
        radius=flat["radius"],
        density=flat["density"],
        poisson=flat["poisson"],
        youngmod=flat["youngmod"],
        fric_rolling=flat["fric_rolling"],
        surf_en_pp=flat["surf_en_pp"],
        fric_dyn_pp=flat["fric_dyn_pp"],
        fric_stat_pp=flat["fric_stat_pp"],
        tan_stiff_r_pp=flat["tan_stiff_r_pp"],
        cor_pp=flat["cor_pp"],
        surf_en_pw=flat["surf_en_pw"],
        fric_dyn_pw=flat["fric_dyn_pw"],
        fric_stat_pw=flat["fric_stat_pw"],
        tan_stiff_r_pw=flat["tan_stiff_r_pw"],
        cor_pw=flat["cor_pw"],
        box_len=flat["box_len"],
        p_compress=flat["p_compress"],
        normal=flat["normal"],
        tangential=flat["tangential"],
        rolling=flat["rolling"],
        adhesion=flat["adhesion"],
        shape=shape,
    )


def iter_space_filling(
    json_path: str,
    factors: list[str],
    bounds: list[tuple[float, float]],
    n_samples: int,
    sampler: qmc.QMCEngine,
) -> list[SimParams]:
    """Expand a space-filling design into a list of :class:`SimParams`.

    Args:
        json_path: Path to a JSON base configuration (OFAT schema — every leaf
            is a single value).
        factors: Names of the continuous parameters to vary.
        bounds: ``(min, max)`` pair for each factor, in the same order.
        n_samples: Number of design points.
        sampler: The QMCEngine instance to use for sampling.

    Returns:
        One :class:`SimParams` per sampled design point.

    Raises:
        ValueError: If ``factors``/``bounds`` lengths differ, a factor is
            unknown or categorical, or any bound is inverted.
    """
    if len(factors) != len(bounds):
        raise ValueError("factors and bounds must have the same length")

    with open(json_path, "r") as f_params:
        params = json.load(f_params, object_pairs_hook=OrderedDict)
    base = resolve_base_params(params)

    unknown = [f for f in factors if f not in base]
    if unknown:
        allowed = [k for k in base if k not in _CATEGORICAL]
        raise ValueError(f"Unknown factors {unknown}. Allowed factors: {allowed}")

    categorical = [f for f in factors if f in _CATEGORICAL]
    if categorical:
        raise ValueError(f"Cannot space-fill categorical factors {categorical}")

    for f, (lo, hi) in zip(factors, bounds):
        if lo >= hi:
            raise ValueError(f"Invalid bounds for '{f}': ({lo}, {hi})")

    samples = sample_space_filling(bounds, n_samples, sampler=sampler)

    all_params = []
    for row in samples:
        flat = dict(base)
        for f, val in zip(factors, row):
            flat[f] = val
        all_params.append(_build_sim_params(flat))
    return all_params


def launch_space_filling(
    sweep_name: str,
    scheduler,
    json_path: str,
    factors: list[str],
    bounds: list[tuple[float, float]],
    n_samples: int,
    sampler: qmc.QMCEngine,
    meshdir: str = "meshes",
    template_dir=None,
    autolaunch: bool = True,
    target: str = "GPU",
    backend: Optional[str] = None,
) -> None:
    """Generate and launch a quasi-Monte Carlo parameter study.

    Example::

        launch_space_filling(
            sweep_name="sobol_study",
            scheduler=RockyScheduler.bb_gpu(),
            json_path="ofat_base.json",
            factors=["cor_pp", "fric_dyn_pp", "youngmod"],
            bounds=[(0.1, 0.9), (0.2, 0.8), (1e6, 1e7)],
            n_samples=32,
            sampler=qmc.Sobol(d=3),
        )

    Args:
        sweep_name: Title of the study, used as the root directory name.
        scheduler: :class:`~rocky_uniaxc.utils.RockyScheduler` for each case.
        json_path: Path to the JSON base configuration.
        factors: Names of the continuous parameters to vary.
        bounds: ``(min, max)`` pair for each factor, in the same order.
        n_samples: Number of design points (cases) to generate.
        sampler: The QMCEngine instance to use for sampling.
        meshdir: Name of the mesh subdirectory inside each case.
        template_dir: Optional directory of custom Jinja2 templates.
        autolaunch: Whether to submit the SLURM jobs after setup.
        target: Compute target — ``"CPU"`` or ``"GPU"``.
        backend: Simulation backend — ``"rocky_prepost"`` or ``"pyrocky"``.

    Raises:
        ValueError: For invalid factors/bounds, backend, or target.
        FileNotFoundError: If ``template_dir`` does not exist.
    """
    all_params = iter_space_filling(
        json_path, factors, bounds, n_samples, sampler=sampler
    )
    launch_param_cases(
        sweep_name,
        scheduler,
        all_params,
        meshdir=meshdir,
        template_dir=template_dir,
        autolaunch=autolaunch,
        target=target,
        backend=backend,
    )
