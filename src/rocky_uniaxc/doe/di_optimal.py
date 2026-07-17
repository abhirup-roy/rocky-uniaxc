"""D- and I-optimal augmentation of an existing DOE."""

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from scipy.stats import qmc
from sklearn.preprocessing import PolynomialFeatures

from ._doe_utils import SimParams, resolve_base_params
from .space_filling import _CATEGORICAL, _SHAPE_FACTORS, _build_sim_params
from .sweep import launch_param_cases
from ..utils import RockyScheduler


_SETTINGS_KEYS = {
    "radius": "p_radius",
    "density": "p_density",
    "poisson": "p_poisson",
    "youngmod": "p_youngmod",
    "box_len": "particle_box_len",
}


def _factor_value(run: SimParams | Mapping, factor: str) -> float:
    if isinstance(run, SimParams):
        owner = run.shape if factor in _SHAPE_FACTORS else run
        return float(getattr(owner, factor))
    key = factor if factor in run else _SETTINGS_KEYS.get(factor, factor)
    try:
        return float(run[key])
    except KeyError as exc:
        raise ValueError(f"Existing run does not contain factor '{factor}'") from exc


def _load_existing_runs(
    existing_runs: str | os.PathLike | Sequence[SimParams | Mapping],
) -> list[SimParams | Mapping]:
    if not isinstance(existing_runs, (str, os.PathLike)):
        return list(existing_runs)

    path = Path(existing_runs)
    settings_paths = (
        [path] if path.is_file() else sorted(path.glob("case_*/settings.json"))
    )
    if not settings_paths:
        raise ValueError(f"No case_*/settings.json files found in {path}")
    return [json.loads(settings.read_text()) for settings in settings_paths]


def iter_di_optimal(
    json_path: str | os.PathLike,
    existing_runs: str | os.PathLike | Sequence[SimParams | Mapping],
    factors: list[str],
    bounds: list[tuple[float, float]],
    n_samples: int,
    criterion: Literal["D", "I", "d", "i"] = "D",
    n_candidates: int = 2048,
    degree: int = 2,
    seed: Optional[int] = None,
) -> list[SimParams]:
    """Choose new runs that optimally augment an existing design.

    Existing runs may be a DOE directory containing ``case_*/settings.json``
    files, or a sequence of :class:`SimParams`/parameter mappings. A quadratic
    response-surface model is used by default. D-optimality maximises parameter
    information; I-optimality minimises average prediction variance over the
    bounded design region.
    """
    criterion = criterion.upper()
    if criterion not in {"D", "I"}:
        raise ValueError("criterion must be 'D' or 'I'")
    if not factors or len(factors) != len(bounds):
        raise ValueError(
            "factors and bounds must be non-empty and have the same length"
        )
    if len(set(factors)) != len(factors):
        raise ValueError("factors must be unique")
    if n_samples < 1 or n_candidates < n_samples:
        raise ValueError("n_samples must be positive and n_candidates >= n_samples")
    if degree < 1:
        raise ValueError("degree must be positive")

    with open(json_path, "r") as f_params:
        base = resolve_base_params(json.load(f_params))
    unknown = [factor for factor in factors if factor not in base]
    if unknown:
        raise ValueError(f"Unknown factors {unknown}")
    categorical = [factor for factor in factors if factor in _CATEGORICAL]
    if categorical:
        raise ValueError(f"Cannot optimise categorical factors {categorical}")

    for factor, (lower, upper) in zip(factors, bounds):
        if lower >= upper:
            raise ValueError(f"Invalid bounds for '{factor}': ({lower}, {upper})")

    runs = _load_existing_runs(existing_runs)
    if not runs:
        raise ValueError("existing_runs must contain at least one run")
    existing = np.array(
        [[_factor_value(run, factor) for factor in factors] for run in runs],
        dtype=float,
    )
    lower = np.array([bound[0] for bound in bounds])
    upper = np.array([bound[1] for bound in bounds])
    if np.any((existing < lower) | (existing > upper)):
        raise ValueError("Existing run values must lie within bounds")

    candidates = qmc.scale(
        qmc.LatinHypercube(d=len(factors), seed=seed).random(n_candidates),
        lower,
        upper,
    )
    if "n_corners" in factors:
        candidates[:, factors.index("n_corners")] = np.rint(
            candidates[:, factors.index("n_corners")]
        )
    candidates = candidates[
        np.all((candidates >= lower) & (candidates <= upper), axis=1)
    ]
    _, unique_idx = np.unique(candidates, axis=0, return_index=True)
    candidates = candidates[np.sort(unique_idx)]
    is_existing = np.any(
        np.all(np.isclose(candidates[:, None], existing[None, :]), axis=2), axis=1
    )
    candidates = candidates[~is_existing]
    if len(candidates) < n_samples:
        raise ValueError("Not enough distinct candidate points; increase n_candidates")

    def normalise(values: np.ndarray) -> np.ndarray:
        return 2 * (values - lower) / (upper - lower) - 1

    model = PolynomialFeatures(degree=degree, include_bias=True)
    existing_model = model.fit_transform(normalise(existing))
    candidate_model = model.transform(normalise(candidates))

    # ponytail: a tiny ridge makes underspecified OFAT starts augmentable; replace
    # with a constrained exchange algorithm only if exact-design benchmarks demand it.
    information = existing_model.T @ existing_model + 1e-9 * np.eye(
        existing_model.shape[1]
    )
    inverse = np.linalg.inv(information)
    moment = candidate_model.T @ candidate_model / len(candidate_model)

    selected = []
    for _ in range(n_samples):
        leverage = np.einsum("ij,jk,ik->i", candidate_model, inverse, candidate_model)
        if criterion == "D":
            scores = leverage
        else:
            reduction = inverse @ moment @ inverse
            scores = np.einsum(
                "ij,jk,ik->i", candidate_model, reduction, candidate_model
            )
            scores /= 1 + leverage

        index = int(np.argmax(scores))
        selected.append(candidates[index])
        row = candidate_model[index]
        direction = inverse @ row
        inverse -= np.outer(direction, direction) / (1 + row @ direction)
        candidates = np.delete(candidates, index, axis=0)
        candidate_model = np.delete(candidate_model, index, axis=0)

    all_params = []
    for point in selected:
        flat = dict(base)
        flat.update(zip(factors, point))
        all_params.append(_build_sim_params(flat))
    return all_params


def launch_di_optimal(
    sweep_name: str,
    scheduler: RockyScheduler,
    json_path: str | os.PathLike,
    existing_runs: str | os.PathLike | Sequence[SimParams | Mapping],
    factors: list[str],
    bounds: list[tuple[float, float]],
    n_samples: int,
    criterion: Literal["D", "I", "d", "i"] = "D",
    n_candidates: int = 2048,
    degree: int = 2,
    seed: Optional[int] = None,
    meshdir: str = "meshes",
    template_dir: Optional[str | os.PathLike] = None,
    autolaunch: bool = True,
    target: str = "GPU",
    backend: Optional[str] = None,
) -> None:
    """Generate and launch D- or I-optimal additions to an existing DOE.

    Selects new design points that augment the supplied runs, creates their
    case directories and simulation inputs, and optionally submits the jobs.

    Example::

        launch_di_optimal(
            sweep_name="d_optimal_extension",
            scheduler=RockyScheduler.bb_gpu(),
            json_path="ofat_base.json",
            existing_runs="ofat_sweep",
            factors=["cor_pp", "fric_dyn_pp", "youngmod"],
            bounds=[(0.1, 0.9), (0.2, 0.8), (1e6, 1e7)],
            n_samples=12,
            criterion="D",
            seed=42,
        )

    ``existing_runs`` accepts three forms. A DOE directory is scanned for
    ``case_*/settings.json`` files. A single settings file may also be passed
    directly; it must contain a numeric value for every requested factor. For
    example, with ``factors=["cor_pp", "fric_dyn_pp", "youngmod"]``::

        {
            "cor_pp": 0.1,
            "fric_dyn_pp": 0.2,
            "p_youngmod": 1000000.0
        }

    Saved settings use ``p_radius``, ``p_density``, ``p_poisson``,
    ``p_youngmod``, and ``particle_box_len`` for the corresponding ``radius``,
    ``density``, ``poisson``, ``youngmod``, and ``box_len`` factors. The factor
    names themselves are also accepted as mapping keys.

    For data already in memory, pass one mapping per existing run::

        existing_runs = [
            {"cor_pp": 0.1, "fric_dyn_pp": 0.2},
            {"cor_pp": 0.5, "fric_dyn_pp": 0.5},
            {"cor_pp": 0.9, "fric_dyn_pp": 0.8},
        ]

    Alternatively, pass a sequence of fully populated :class:`SimParams`
    objects::

        from rocky_uniaxc.doe.sweep import iter_params

        existing_runs = iter_params("previous_sweep.json")

    Only the attributes named in ``factors`` are read from each object.

    Args:
        sweep_name: Title of the study, used as the root directory name.
        scheduler: :class:`~rocky_uniaxc.utils.RockyScheduler` for each case.
        json_path: Path to the JSON base configuration.
        existing_runs: Existing design points in one of the directory, JSON,
            mapping-sequence, or :class:`SimParams`-sequence forms described
            above.
        factors: Names of the continuous parameters to optimise.
        bounds: ``(min, max)`` pair for each factor, in the same order.
        n_samples: Number of additional design points (cases) to generate.
        criterion: Optimality criterion. ``"D"`` maximises parameter
            information; ``"I"`` minimises average prediction variance.
            Matching lowercase values are also accepted.
        n_candidates: Number of Latin-hypercube candidate points considered
            during optimisation.
        degree: Degree of the polynomial response-surface model.
        seed: Optional random seed for reproducible candidate generation.
        meshdir: Name of the mesh subdirectory inside each case.
        template_dir: Optional directory of custom Jinja2 templates.
        autolaunch: Whether to submit the SLURM jobs after setup.
        target: Compute target — ``"CPU"`` or ``"GPU"``.
        backend: Simulation backend — ``"rocky_prepost"`` or ``"pyrocky"``.

    Raises:
        ValueError: For invalid design inputs, backend, or target.
        FileNotFoundError: If ``json_path`` or ``template_dir`` does not exist.
        NotImplementedError: If ``target="MULTI_GPU"`` is requested.
    """
    all_params = iter_di_optimal(
        json_path,
        existing_runs,
        factors,
        bounds,
        n_samples,
        criterion=criterion,
        n_candidates=n_candidates,
        degree=degree,
        seed=seed,
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
