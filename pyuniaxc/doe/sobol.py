import json
import inspect
import os
import subprocess
from collections import OrderedDict
import pandas as pd
from typing import Callable, Optional

from scipy.stats import qmc
import numpy as np
import jinja2
from ..utils import slurm_sbatch, cd
from ..compr_meshgen import create_meshes_efficiently


class ConstrainedSobol:
    """
    A wrapper for scipy.stats.qmc.Sobol that applies a custom transformation
    to generate samples satisfying specific constraints.

    This class uses composition to manage a Sobol sampler and applies a
    user-provided transformer function to map points from the unit hypercube
    to a constrained, scaled domain.
    """

    def __init__(
        self,
        d: int,
        bounds: dict[str, list[float]],
        transformer: Callable[[np.ndarray, dict[str, list[float]]], np.ndarray],
        scramble: bool = True,
        bits: Optional[int] = None,
        seed: int = 42,
        optimization: Optional[str] = None,
    ):
        """
        Initialises the ConstrainedSobol sampler.

        Args:
            d (int): Number of dimensions (variables).
            bounds (Dict[str, List[float]]): A dictionary mapping variable names
                to their [min, max] bounds.
            transformer (Callable): A function that takes a (n, d) numpy array of
                points in the unit cube and the bounds dict, and returns the
                (n, d) transformed, constrained points.
            scramble (bool): Whether to use a scrambled Sobol sequence.
            bits (int, optional): Number of bits for the Sobol sequence. From `scipy.stats.qmc.Sobol`.
            seed (int, optional): Seed for the random number generator.
            optimization (str, optional): Optimization method for the Sobol sequence. From `scipy.stats.qmc.Sobol`.
        """
        if d != len(bounds):
            raise ValueError(
                "Dimension 'd' must be equal to the number of bounds provided."
            )

        self.d = d
        self.bounds = bounds
        self.transformer = transformer

        sobol_kwargs = inspect.getfullargspec(qmc.Sobol.__init__).kwonlyargs
        if "seed" in sobol_kwargs:
            self.sampler = qmc.Sobol(
                d=d, scramble=scramble, bits=bits, seed=seed, optimization=optimization
            )
        elif "rng" in sobol_kwargs:
            rng = np.random.default_rng(seed)
            self.sampler = qmc.Sobol(
                d=d, scramble=scramble, bits=bits, rng=rng, optimization=optimization
            )
        else:
            self.sampler = qmc.Sobol(
                d=d, scramble=scramble, bits=bits, optimization=optimization
            )

    def _store_unit_cube(self, constrained_cube: np.ndarray) -> None:
        scaler = MinMaxScaler()
        scaler.fit(constrained_cube)
        self.constrained_unit_cube = scaler.transform(constrained_cube)

    def random(self, n: int) -> np.ndarray:
        """
        Generate 'n' constrained quasi-random samples.

        Args:
            n (int): The number of samples to generate.

        Returns:
            np.ndarray: An array of shape (n, d) containing the samples.
        """
        points_unit_cube = self.sampler.random(n=n)
        constrained_points = self.transformer(points_unit_cube, self.bounds)
        self._store_unit_cube(constrained_points)

        return constrained_points

    def random_base2(
        self, min_N: Optional[int] = None, m: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate 2^m samples, where m is the smallest integer such that 2^m >= min_N.

        Args:
            min_N (int, optional): Minimum number of samples to generate.
            m (int, optional): If provided, generates exactly 2^m samples.

        """
        if min_N:
            m = int(np.ceil(np.log2(min_N)))
        elif m:
            m = m
        else:
            raise ValueError("Either min_N or m must be provided.")

        points_unit_cube = self.sampler.random_base2(m=m)
        constrained_points = self.transformer(points_unit_cube, self.bounds)
        self._store_unit_cube(constrained_points)

        return constrained_points


class LShapedTransformer:
    """
    A configurable transformer class for a 2D L-shaped constraint.

    Upon initialization, it takes the names of the two variables to be
    constrained. The instance can then be passed to the ConstrainedSobol class.
    """

    def __init__(self, constrained_vars: list[str]):
        """
        Initializes the transformer.

        Args:
            constrained_vars (List[str]): A list containing the names of the
                two variables to be constrained, e.g., ['Pressure', 'Temperature'].
        """
        if len(constrained_vars) != 2:
            raise ValueError("LShapedTransformer requires exactly two variable names.")
        self.constrained_vars = constrained_vars

    def __call__(
        self, points_unit_cube: np.ndarray, bounds: dict[str, list[float]]
    ) -> np.ndarray:
        """
        Makes the instance callable, performing the transformation.
        """
        _, d = points_unit_cube.shape
        var_names = list(bounds.keys())

        # --- 1. Create a dynamic mapping from variable name to column index ---
        name_to_idx = {name: i for i, name in enumerate(var_names)}

        # --- 2. Identify indices for constrained and unconstrained variables ---
        idx1 = name_to_idx[self.constrained_vars[0]]
        idx2 = name_to_idx[self.constrained_vars[1]]

        unconstrained_indices = [i for i in range(d) if i not in [idx1, idx2]]

        # --- 3. Calculate area ratio for partitioning ---
        b_constr1 = bounds[self.constrained_vars[0]]
        b_constr2 = bounds[self.constrained_vars[1]]

        area_A = (b_constr1[1] - b_constr1[0]) * (1.0 - b_constr2[0])
        area_B = (1.0 - b_constr1[0]) * (b_constr2[1] - 1.0)
        split_ratio = area_A / (area_A + area_B)

        final_samples = np.zeros_like(points_unit_cube)
        u1, u2 = (
            points_unit_cube[:, 0],
            points_unit_cube[:, 1],
        )  # Use first 2 Sobol dims

        mask_A = u1 < split_ratio
        mask_B = ~mask_A

        # Helper for scaling
        def scale_val(val, min_b, max_b):
            return min_b + val * (max_b - min_b)

        # Calculate the transformed values for the two constrained variables
        x_constr1_A = scale_val(u2[mask_A], b_constr1[0], b_constr1[1])
        x_constr2_A = scale_val(u1[mask_A] / split_ratio, b_constr2[0], 1.0)

        x_constr1_B = scale_val(u2[mask_B], b_constr1[0], 1.0)
        x_constr2_B = scale_val(
            (u1[mask_B] - split_ratio) / (1 - split_ratio), 1.0, b_constr2[1]
        )

        # Assign the transformed values to their designated columns
        final_samples[mask_A, idx1] = x_constr1_A
        final_samples[mask_A, idx2] = x_constr2_A
        final_samples[mask_B, idx1] = x_constr1_B
        final_samples[mask_B, idx2] = x_constr2_B

        for i, u_idx in enumerate(range(2, d)):
            target_idx = unconstrained_indices[i]
            var_name = var_names[target_idx]
            b = bounds[var_name]
            u = points_unit_cube[:, u_idx]
            final_samples[:, target_idx] = scale_val(u, b[0], b[1])

        return final_samples


def iter_sobol(
    json_path: str, sobol_values: dict[str, list | float | type], min_N: int = 300
):
    pass
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
    if not isinstance(sobol_values, dict):
        raise ValueError("OFAT values must be provided as a dictionary.")

    ofat_dict_check = set(sobol_values.keys()) == set(
        ["parameters", "test_range", "dtype"]
    )
    if not ofat_dict_check:
        raise ValueError(
            "OFAT values must contain 'parameters', 'test_range', 'dtype' keys."
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

    if not set(sobol_values["parameters"]).issubset(set(ofat_base_valid.keys())):
        raise ValueError(
            f"Invalid OFAT parameters. Allowed parameters are: {list(ofat_base_valid.keys())}"
        )

    # TODO: Double check this
    range_valid = {
        "fric_dyn_pp": (0.0, None),
        "fric_stat_pp": (0.0, None),
        "fric_rolling_pp": (0.0, None),
        "cor_pp": (0.0, 1.0),
        "fric_dyn_pw": (0.0, None),
        "fric_stat_pw": (0.0, None),
        "cor_pw": (0.0, 1.0),
        "box_len": (0.0, None),
        "vert_ar": (0.0, None),
        "horiz_ar": (0.0, None),
        "n_corners": (10, 30),
        "sq_degree": (2.0, 12.0),
    }

    if "vert_ar" and "horiz_ar" in sobol_values["parameters"]:
        domain_transformer = LShapedTransformer(
            constrained_vars=["vert_ar", "horiz_ar"]
        )
        c_sampler = ConstrainedSobol(
            d=len(sobol_values["parameters"]),
            bounds={
                p: r
                for p, r in zip(sobol_values["parameters"], sobol_values["test_range"])
            },
            transformer=domain_transformer,
        )

        # Min. for L-shape constraint
        samples = c_sampler.random_base2(min_N=min_N)
    else:
        # --- Unconstrained sampling ---
        sampler = qmc.Sobol(d=len(sobol_values["parameters"]), scramble=True)
        samples_unit = sampler.random(n=np.log2(min_N).ceil().astype(int))

        # Scale samples to the specified ranges
        l_bounds = [r[0] for r in sobol_values["test_range"]]
        u_bounds = [r[1] for r in sobol_values["test_range"]]
        samples = qmc.scale(samples_unit, l_bounds, u_bounds)

    experiments_df = pd.DataFrame(samples, columns=sobol_values["parameters"])
    # Apply datatype corrections
    for i, col in enumerate(sobol_values["parameters"]):
        experiments_df[col] = experiments_df[col].astype(sobol_values["dtype"][i])

    return experiments_df, ofat_base_valid


def launch_sobol(
    sweep_name: str,
    autolaunch: bool,
    json_path: str,
    sobol_values: dict[str, list | str],
    loc: str = "bb-cpu",
    target: str = "CPU",
    n_points: int = 10,
    ncpus: Optional[int] = None,
    ngpus: Optional[int] = None,
    run_days: int = 10,
    **kwargs,
):
    custom_sh = kwargs.get("custom_sh")
    if not (template_dir := kwargs.get("template_dir")):
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
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
    rocky_templ_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(f"{template_dir}")
    )
    rocky_template = rocky_templ_env.get_template("template_uniax.py")

    experiments_df, base_dict = iter_sobol(
        json_path=json_path, sobol_values=sobol_values, min_N=n_points
    )

    total_cases = len(experiments_df)
    vars = experiments_df.columns.tolist()

    # Generate all cases
    all_params = []

    for _, row in experiments_df.iterrows():
        case_params = base_dict.copy()
        for v in vars:
            case_params[v] = row[v]
        all_params.append(case_params)

    os.makedirs(sweep_name, exist_ok=True)
    case_dirs = []
    for i in range(total_cases):
        case_dir = os.path.join(sweep_name, f"case_{i}")
        os.makedirs(case_dir, exist_ok=True)
        os.makedirs(os.path.join(case_dir, "plots"), exist_ok=True)
        case_dirs.append(case_dir)

    if "box_len" in vars:
        unique_sizes = set(experiments_df["box_len"])
    elif "box_len" in base_dict:
        unique_sizes = {base_dict["box_len"]}
    else:
        raise ValueError(
            "No box length parameter found in experiments or base dictionary. \n"
            "Debugging required."
        )

    size_to_mesh_dir = {}
    for size in unique_sizes:
        shared_mesh_dir = os.path.join(sweep_name, f"meshes_{size}")
        os.makedirs(shared_mesh_dir, exist_ok=True)

        create_meshes_efficiently(size=size, meshsize=0.01, out_dir=shared_mesh_dir)
        size_to_mesh_dir[size] = shared_mesh_dir

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
        print("Launching all cases...")
        for i, case_dir in enumerate(case_dirs):
            print(f"Launching case {i}/{total_cases}...")
            # Use subprocess to launch in the background
            with cd(case_dir):
                try:
                    result = subprocess.run(
                        ["sbatch", "runRocky.sh"],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    print(f"Job submitted: {result.stdout.strip()}")
                except subprocess.CalledProcessError as e:
                    print(f"Error submitting job: {e.stderr}")

    print(f"All {total_cases} cases prepared and launched.")
    print("Exiting launcher script now")
