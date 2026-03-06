import time
import os
import pathlib
import shutil
import subprocess
from typing import Optional
from dataclasses import dataclass, asdict, field
import json

import numpy as np
import matplotlib.pyplot as plt
import ansys.rocky.core as rocky_api

from . import particles_shapes
from .compr_meshgen import create_meshes_efficiently

__all__ = ["Parameters", "UniaxialCompressionSimulation"]

@dataclass(slots=True)
class Parameters:
    project_dir: str | pathlib.Path

    particle_box_len: float
    t_fill: float
    t_settle: float
    t_compress: float
    p_compress: float

    p_radius: float | dict
    p_density: float
    p_youngmod: float
    p_poisson: float
    fric_dyn_pp: float
    fric_stat_pp: float
    cor_pp: float
    fric_dyn_pw: float
    fric_stat_pw: float
    cor_pw: float

    normal_force_model: str = (
        "linear_hysteresis"  # highly suitable for uniaxial compression
    )
    tangential_force_model: str = "coulomb_limit"
    adhesion_model: str = "none"  # 'none' 'constant' 'linear' or 'JKR'
    # Rolling friction off by default, unreliable for polyhedra
    rolling_fric: float = 0.0
    rolling_model: str = "none"
    neighbor_search: str = "BVH"  # 'BVH' 'RegularGrid' or 'SparseGrid'
    processor: str = "GPU"  # 'CPU' or 'GPU'

    mesh_dir: Optional[str | pathlib.Path] = None
    plots_dir: Optional[str | pathlib.Path] = None

    shape_name: str = (
        "sphere"  # 'sphere', 'polyhedron', 'sphero_cylinder', or 'custom_polyhedron'
    )
    vert_ar: float = 1.0  # vertical aspect ratio for shaped particles
    horiz_ar: float = 1.0  # horizontal aspect ratio for shaped particles
    n_corners: int = 30  # number of corners for polyhedral particles
    sq_degree: float = 2.0  # superquadric degree for shaped particles
    particle_path: Optional[str] = (
        None  # path to custom particle STL file, required if shape_name is 'custom_polyhedron'
    )
    smoothness: Optional[float] = None

    shape_dict: dict[str, (str | int | float | None)] = field(default_factory=dict)

    def __post_init__(self):
        if self.mesh_dir is None:
            self.mesh_dir = (
                pathlib.Path(self.project_dir).parent
                / f"meshes_{self.particle_box_len}"
            )
        else:
            self.mesh_dir = pathlib.Path(self.mesh_dir)

        if not self.mesh_dir.exists():
            create_meshes_efficiently(
                size=self.particle_box_len,
                out_dir=self.mesh_dir,
            )

        if not self.plots_dir:
            self.plots_dir = pathlib.Path(self.project_dir).parent / "plots"
        else:
            self.plots_dir = pathlib.Path(self.plots_dir)

        self.project_dir = pathlib.Path(self.project_dir)

        self.shape_dict["name"] = self.shape_name
        self.shape_dict["vert_ar"] = self.vert_ar
        self.shape_dict["horiz_ar"] = self.horiz_ar
        self.shape_dict["n_corners"] = self.n_corners
        self.shape_dict["sq_degree"] = self.sq_degree
        self.shape_dict["smoothness"] = self.smoothness
        self.shape_dict["particle_path"] = self.particle_path
        self._validate()

    def _validate(self):
        errors = []

        # --- Positive floats ---
        positive_fields = {
            "particle_box_len": self.particle_box_len,
            "t_fill": self.t_fill,
            "t_settle": self.t_settle,
            "t_compress": self.t_compress,
            "p_compress": self.p_compress,
            "p_density": self.p_density,
            "p_youngmod": self.p_youngmod,
        }
        for name, val in positive_fields.items():
            if val <= 0:
                errors.append(f"'{name}' must be > 0, got {val}.")

        # --- Bounded [0, 1] fields ---
        unit_fields = {
            "p_poisson": self.p_poisson,
            "cor_pp": self.cor_pp,
            "cor_pw": self.cor_pw,
        }
        for name, val in unit_fields.items():
            if not (0.0 <= val <= 1.0):
                errors.append(f"'{name}' must be in [0, 1], got {val}.")

        # --- Non-negative floats ---
        nonneg_fields = {
            "fric_dyn_pp": self.fric_dyn_pp,
            "fric_stat_pp": self.fric_stat_pp,
            "fric_dyn_pw": self.fric_dyn_pw,
            "fric_stat_pw": self.fric_stat_pw,
            "rolling_fric": self.rolling_fric,
            "vert_ar": self.vert_ar,
            "horiz_ar": self.horiz_ar,
        }
        for name, val in nonneg_fields.items():
            if val < 0:
                errors.append(f"'{name}' must be >= 0, got {val}.")

        # --- p_radius ---
        if isinstance(self.p_radius, float):
            if self.p_radius <= 0:
                errors.append(f"'p_radius' must be > 0, got {self.p_radius}.")
        elif isinstance(self.p_radius, dict):
            if not self.p_radius:
                errors.append("'p_radius' dict must not be empty.")
            else:
                if any(r <= 0 for r in self.p_radius.keys()):
                    errors.append("All radii in 'p_radius' dict must be > 0.")
                prob_sum = sum(self.p_radius.values())
                if not (np.isclose(prob_sum, 1.0) or np.isclose(prob_sum, 100.0)):
                    errors.append(
                        f"'p_radius' probabilities must sum to 1 or 100, got {prob_sum}."
                    )
        else:
            errors.append(
                f"'p_radius' must be a float or dict, got {type(self.p_radius)}."
            )

        # --- Enum-like string fields ---
        valid_normal = {"linear_hysteresis", "hertz", "linear_spring"}
        if self.normal_force_model not in valid_normal:
            errors.append(
                f"'normal_force_model' must be one of {valid_normal}, "
                f"got '{self.normal_force_model}'."
            )

        valid_tangential = {"coulomb_limit", "linear_spring_coulomb_limit"}
        if self.tangential_force_model not in valid_tangential:
            errors.append(
                f"'tangential_force_model' must be one of {valid_tangential}, "
                f"got '{self.tangential_force_model}'."
            )

        valid_adhesion = {"none", "constant", "linear", "JKR"}
        if self.adhesion_model not in valid_adhesion:
            errors.append(
                f"'adhesion_model' must be one of {valid_adhesion}, "
                f"got '{self.adhesion_model}'."
            )

        valid_rolling = {"none", "type_a", "type_b"}
        if self.rolling_model not in valid_rolling:
            errors.append(
                f"'rolling_model' must be one of {valid_rolling}, "
                f"got '{self.rolling_model}'."
            )

        valid_neighbor = {"BVH", "RegularGrid", "SparseGrid"}
        if self.neighbor_search not in valid_neighbor:
            errors.append(
                f"'neighbor_search' must be one of {valid_neighbor}, "
                f"got '{self.neighbor_search}'."
            )

        valid_processor = {"CPU", "GPU"}
        if self.processor not in valid_processor:
            errors.append(
                f"'processor' must be one of {valid_processor}, "
                f"got '{self.processor}'."
            )

        valid_shapes = {"sphere", "polyhedron", "sphero_cylinder", "custom_polyhedron"}
        if self.shape_name not in valid_shapes:
            errors.append(
                f"'shape_name' must be one of {valid_shapes}, "
                f"got '{self.shape_name}'."
            )

        if self.shape_name == "custom_polyhedron":
            if not self.particle_path:
                errors.append(
                    "'particle_path' must be provided when shape_name is 'custom_polyhedron'."
                )
            elif not pathlib.Path(self.particle_path).is_file():
                errors.append(
                    f"'particle_path' does not point to a valid file: {self.particle_path}"
                )

        if self.n_corners < 10:
            errors.append(f"'n_corners' must be >= 10, got {self.n_corners}.")

        if self.sq_degree < 2.0:
            errors.append(f"'sq_degree' must be >= 2.0, got {self.sq_degree}.")

        if errors:
            raise ValueError(
                "Invalid Parameters:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    @classmethod
    def from_json(cls, path: str | pathlib.Path, project_dir: str | pathlib.Path) -> "Parameters":
        with open(path, "r") as f:
            data = json.load(f)

        shape = data["shape"]
        props = data["particle_properties"]
        inter = data["inseractions"]  # note: typo in JSON preserved
        exp = data["experim_settings"]
        contact = data["contact_model"]

        return cls(
            project_dir=project_dir,
            # Particle shape
            shape_name=shape["name"],
            vert_ar=shape.get("vert_ar", 1.0),
            horiz_ar=shape.get("horiz_ar", 1.0),
            n_corners=shape.get("n_corners", 30),
            sq_degree=shape.get("sq_degree", 2.0),
            # Particle properties
            p_radius=props["radius"],
            p_density=props["density"],
            p_poisson=props["poisson"],
            p_youngmod=props["youngmod"],
            # Interactions
            fric_dyn_pp=inter["pp"]["fric_dyn"],
            fric_stat_pp=inter["pp"]["fric_stat"],
            cor_pp=inter["pp"]["cor"],
            rolling_fric=inter["pp"].get("fric_rolling", 0.0),
            fric_dyn_pw=inter["pw"]["fric_dyn"],
            fric_stat_pw=inter["pw"]["fric_stat"],
            cor_pw=inter["pw"]["cor"],
            # Experiment settings
            particle_box_len=exp["box_len"],
            p_compress=exp["p_compress"],
            t_fill=exp.get("t_fill", 1.0),
            t_settle=exp.get("t_settle", 0.5),
            t_compress=exp.get("t_compress", 2.0),
            # Contact models
            normal_force_model=contact["normal"],
            tangential_force_model=contact["tangential"],
            rolling_model=contact["rolling"],
            adhesion_model=contact["adhesion"],
        )

class UniaxialCompressionSimulation:
    def __init__(
        self,
        params: Parameters,
        rocky_exe_path: Optional[str] = None,
        insertion=True,
        filename: str = "uniaxial_compression.rocky",
        headless: bool = True,
    ):
        self.params = params
        self.insertion = insertion
        self.filename = filename
        self.headless = headless

        if not rocky_exe_path:
            rocky_exe_path = shutil.which("Rocky")
            if not rocky_exe_path:
                raise FileNotFoundError(
                    "Rocky executable not found in system PATH. \n"
                    "Please provide the path to the Rocky executable."
                )
        elif not pathlib.Path(rocky_exe_path).is_file():
            raise FileNotFoundError(
                f"Provided Rocky executable path is invalid: {rocky_exe_path}"
            )
        self.rocky_exe_path = rocky_exe_path
        self.setup()

        self._particle = None
        self._mesh = {}
        self._materials = {}

        self.active_boxes = {}
        self.active_euls = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.rocky.close()
        return False

    def setup(self):
        try:
            self.rocky = rocky_api.launch_rocky(self.rocky_exe_path, headless=self.headless)
        except Exception as e:
            raise e

        self._project = self.rocky.api.CreateProject()
        self._project.SaveProject(
            str(pathlib.Path(self.params.project_dir) / self.filename)
        )
        self._study = self._project.GetStudy()
        self._study.SetName("Uniaxial Compression")

    def load_meshes(self, insert=True):
        assert self.params.mesh_dir is not None
        mesh_dir = pathlib.Path(self.params.mesh_dir).resolve()

        top_wall_path = mesh_dir / "compressive_wall1.stl"
        top_wall = self._study.ImportWall(
            str(top_wall_path), import_scale=1.1, convert_yz=False
        )[0]
        top_wall.SetName("Top Wall")
        top_wall.SetBoundaryMass(1e-6)
        top_wall.SetTranslation([0, self.params.particle_box_len / 2 + 1e-6, 0])

        bottom_wall_path = mesh_dir / "compressive_wall2.stl"
        bottom_wall = self._study.ImportWall(
            str(bottom_wall_path), import_scale=1.1, convert_yz=False
        )[0]
        bottom_wall.SetName("Bottom Wall")

        # Store proxies directly — only used within same session, never passed cross-IPC
        self._mesh["top_wall"] = top_wall
        self._mesh["bottom_wall"] = bottom_wall

        if insert:
            insert_stl_path = (mesh_dir / "insert.stl").resolve()
            insert_inlet = self._study.ImportSurface(
                str(insert_stl_path), import_scale=1.0, convert_yz=True
            )[0]
            insert_inlet.SetName("Insert Inlet")  # <-- set explicit name
            insert_inlet.SetPivotPoint([0, 0, 0])

            current_height = insert_inlet.GetVertices().mean(axis=0)[1]
            target_height = (self.params.particle_box_len / 2) * 0.99

            insert_inlet.SetTranslation([0, float(target_height - current_height), 0])
            insert_inlet.SetInvertNormal(True)
            self._mesh["insert_inlet"] = insert_inlet

    def load_material_properties(self):
        material_collection = self._study.GetMaterialCollection()

        particle_mat = material_collection.AddSolidMaterial()
        particle_mat.SetName("Particle Material")
        particle_mat.SetDensity(self.params.p_density)
        particle_mat.SetYoungsModulus(self.params.p_youngmod)
        particle_mat.SetPoissonRatio(self.params.p_poisson)
        particle_mat.SetUseBulkDensity(False)

        wall_mat = material_collection.AddSolidMaterial()
        wall_mat.SetName("Wall Material")
        wall_mat.SetDensity(2700)
        wall_mat.SetYoungsModulus(1e9)
        wall_mat.SetPoissonRatio(0.3)
        wall_mat.SetUseBulkDensity(False)

        # Introspect before using
        import inspect
        from ansys.rocky.core.rocky_api_proxies import ApiElementProxy
        print("serialize signature:", inspect.signature(ApiElementProxy.serialize))
        print("wall_mat type:", type(wall_mat))
        print("wall_mat._pool_id:", wall_mat._pool_id)

        # Always use name strings — never pass proxies as arguments
        self._mesh["top_wall"].SetMaterial(wall_mat.serialize(wall_mat))
        self._mesh["bottom_wall"].SetMaterial(wall_mat.serialize(wall_mat))
        self._materials["particle_mat"] = particle_mat
        self._materials["wall_mat"] = wall_mat

    def load_interactions(self):
        pm = self._materials["particle_mat"]
        wm = self._materials["wall_mat"]

        interaction_collection = self._study.GetMaterialsInteractionCollection()
        pp_interaction = interaction_collection.GetMaterialsInteraction(
            pm.serialize(pm), pm.serialize(pm)
        )
        pw_interaction = interaction_collection.GetMaterialsInteraction(
            pm.serialize(pm), wm.serialize(wm)
        )

        pp_interaction.SetRestitutionCoefficient(self.params.cor_pp)
        pp_interaction.SetDynamicFriction(self.params.fric_dyn_pp)
        pp_interaction.SetStaticFriction(self.params.fric_stat_pp)

        pw_interaction.SetRestitutionCoefficient(self.params.cor_pw)
        pw_interaction.SetDynamicFriction(self.params.fric_dyn_pw)
        pw_interaction.SetStaticFriction(self.params.fric_stat_pw)

    def gen_particle(self):
        shape_name = self.params.shape_dict["name"]
        self._particle = self._study.CreateParticle()
        self._particle.SetName("Particle")

        match shape_name:
            case "sphere":
                shape = particles_shapes.Sphere(radius=self.params.p_radius)
            case "polyhedron":
                shape = particles_shapes.Polyhedron(
                    radius=self.params.p_radius,
                    vert_ar=self.params.vert_ar,
                    horiz_ar=self.params.horiz_ar,
                    n_corners=self.params.n_corners,
                    superquadric_degree=self.params.sq_degree,
                )
            case "sphero_cylinder":
                shape = particles_shapes.SpheroCylinder(
                    radius=self.params.p_radius, vert_ar=self.params.vert_ar
                )
            case "custom_polyhedron":
                if (
                    not self.params.particle_path
                    or not pathlib.Path(self.params.particle_path).is_file()
                ):
                    raise ValueError(
                        "Particle path must be provided for custom polyhedron shape."
                    )
                shape = particles_shapes.CustomPolyhedron(
                    stl_path=self.params.particle_path, radius=self.params.p_radius
                )
            case _:
                raise ValueError(
                    f"Unsupported shape type: {shape_name}"
                    "Supported shapes are: 'sphere', 'polyhedron', 'sphero_cylinder', and 'custom_polyhedron'."
                )
        pm = self._materials["particle_mat"]
        shape.particle2rocky(
            particle=self._particle,
            material=pm.serialize(pm),
            rolling_friction=self.params.rolling_fric,
        )

    def sim_physics(self):
        physics = self._study.GetPhysics()
        physics.SetNormalForceModel(self.params.normal_force_model)
        physics.SetTangentialForceModel(self.params.tangential_force_model)
        physics.SetAdhesionModel(self.params.adhesion_model)

        physics.SetGravityXDirection(0)
        physics.SetGravityYDirection(-9.81)
        physics.SetGravityZDirection(0)

    def insertion_settings(self, insert=True):

        fill_box_vol = self.params.particle_box_len**3
        if isinstance(self.params.p_radius, float):
            particle_vol = (4 / 3) * np.pi * self.params.p_radius**3
        elif isinstance(self.params.p_radius, dict):
            radii = np.array(list(self.params.p_radius.keys()))
            probs = np.array(list(self.params.p_radius.values()))
            probs /= probs.sum()  # normalize probabilities
            avg_radius = np.sum(radii * probs)
            particle_vol = (4 / 3) * np.pi * avg_radius**3
        else:
            raise ValueError(
                "p_radius must be either a float or a dict of {radius: probability}."
                " If using a dict, values must sum to 1 or 100"
                f"Received: {type(self.params.p_radius)}"
            )

        n_particles = (
            np.rint(fill_box_vol / particle_vol * 0.5).astype(int).item()
        )  # target 50% fill
        mass_particles = particle_vol * self.params.p_density * n_particles

        if insert:
            inlet = self._mesh["insert_inlet"]
            particle_inlet = self._study.CreateParticleInlet(
                inlet.serialize(inlet),
                self._particle.serialize(self._particle),
            )
            flowr = mass_particles / self.params.t_fill

            input_property_lst = particle_inlet.GetInputPropertiesList()
            input_property_lst[0].SetMassFlowRate(flowr, "kg/s")

            particle_inlet.SetStartTime(0.0, "s")
            particle_inlet.SetStopTime(self.params.t_fill, "s")
            particle_inlet.DisablePeriodic()
        else:
            raise NotImplementedError(
                "Volumetric insertion is not yet implemented."
                "Raise an issue if you would like to see this feature added."
            )

    def move_top_wall(self, insert=True):
        frame_source = self._study.GetMotionFrameSource()
        top_wall_frame = frame_source.NewFrame()

        motions = top_wall_frame.GetMotions()

        # drop almost weightless wall
        drop_wall_motion = motions.New()
        drop_wall_motion.SetType("Free Body Translation")
        free_body = drop_wall_motion.GetTypeObject()
        free_body.SetFreeMotionDirection("y")
        drop_wall_motion.SetStartTime(
            self.params.t_fill + self.params.t_settle
            if insert
            else self.params.t_settle
        )

        f_compr = 1e-6 * 9.81 - self.params.p_compress * self.params.particle_box_len**2
        compr_motion = motions.New()
        compr_motion.SetType("Additional Force")
        add_force = compr_motion.GetTypeObject()
        add_force.SetForceValue([0, f_compr, 0])

        if insert:
            start_time = self.params.t_fill + self.params.t_settle + 0.1
        else:
            start_time = self.params.t_settle + 0.1
        end_time = start_time + self.params.t_compress
        compr_motion.SetStartTime(start_time)
        compr_motion.SetStopTime(end_time)

        top_wall_frame.ApplyTo(self._mesh["top_wall"].serialize(self._mesh["top_wall"]))

    def set_domain_settings(self):
        domain_settings = self._study.GetDomainSettings()
        domain_settings.DisableUseBoundaryLimits()
        domain_settings.DisablePeriodicAtGeometryLimits()

        domain_settings.SetDomainType("CARTESIAN")
        domain_settings.SetCoordinateLimitsMinValues(
            [
                (-self.params.particle_box_len / 2) * 1.5,
                (-self.params.particle_box_len / 2) * 1.5,
                (-self.params.particle_box_len / 2) * 1.5,
            ]
        )
        domain_settings.SetCoordinateLimitsMaxValues(
            [
                (self.params.particle_box_len / 2) * 1.5,
                (self.params.particle_box_len / 2) * 1.5,
                (self.params.particle_box_len / 2) * 1.5,
            ]
        )

        domain_settings.SetCartesianPeriodicDirections("XZ")
        domain_settings.SetPeriodicLimitsMinCoordinates(
            [
                -self.params.particle_box_len / 2,
                -1e-6,
                -self.params.particle_box_len / 2,
            ]
        )
        domain_settings.SetPeriodicLimitsMaxCoordinates(
            [
                self.params.particle_box_len / 2,
                1e-6,
                self.params.particle_box_len / 2,
            ]
        )

    def _check_nvidia_gpu(self):
        try:
            output = subprocess.check_output(["nvidia-smi", "-L"], encoding="utf-8")
            count = len([line for line in output.strip().split("\n") if line])
            return count

        except (subprocess.CalledProcessError, FileNotFoundError):
            return 0

    def _select_processor(self, solver):
        if self.params.processor == "GPU":
            if not (n_gpus := self._check_nvidia_gpu()):
                print("Warning: No NVIDIA GPU detected. Falling back to CPU.")
                solver.SetSimulationTarget("CPU")
            else:
                if n_gpus >= 1:
                    solver.SetSimulationTarget("GPU")
                # TODO: Add support for multi-GPU setups

        elif self.params.processor == "CPU":
            solver.SetSimulationTarget("CPU")

            cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count() or 1))
            solver.SetNumberOfProcessors(cpus)

    def load_modules(self):
        contacts_data = self._study.GetContactData()
        contacts_data.EnableCollectContactsData()
        if self.params.adhesion_model != "none":
            contacts_data.EnableIncludeAdhesiveContacts()

    def simulate(self, insert=True):
        solver = self._study.GetSolver()
        self._select_processor(solver)

        if insert:
            runtime = sum(
                [self.params.t_fill, self.params.t_settle, self.params.t_compress]
            )
        else:
            runtime = sum([self.params.t_settle, self.params.t_compress])
        solver.SetSimulationDuration(runtime, "s")

        self._project.SaveProject()

        print(f"Starting simulation with {solver.GetSimulationTarget()} solver...")
        self._study.StartSimulation(non_blocking=True)

        while self._study.IsSimulating():
            self._study.RefreshResults()
            print(f"Simulation Progress: {self._study.GetProgress():.2f} %")

            time.sleep(2)

        print("Simulation completed.")

    def _get_cropped_region(self, particles, time_step, sample_frac=0.9):
        if time_step in self.active_boxes:
            return self.active_boxes[time_step]

        x_coords = particles.GetGridFunction("Coordinate : X").GetArray(
            time_step=time_step
        )
        y_coords = particles.GetGridFunction("Coordinate : Y").GetArray(
            time_step=time_step
        )
        z_coords = particles.GetGridFunction("Coordinate : Z").GetArray(
            time_step=time_step
        )

        positions = np.vstack((x_coords, y_coords, z_coords))
        pos_rngs = np.ptp(positions, axis=1)
        sample_rng = pos_rngs * sample_frac

        processes = self._project.GetUserProcessCollection()

        cube_selection = processes.CreateCubeProcess(particles)
        cube_selection.SetCenter(x_coords.mean(), y_coords.mean(), z_coords.mean())
        cube_selection.SetSize(sample_rng[0], sample_rng[1], sample_rng[2])

        self.active_boxes[time_step] = cube_selection

        return cube_selection

    def _calc_bulk_density(self, particles, time_step, sample_frac=0.9):

        cube_selection = self._get_cropped_region(particles, time_step, sample_frac)

        mass_arr = cube_selection.GetGridFunction("Particle Mass").GetArray(
            time_step=time_step
        )
        sample_mass = mass_arr.sum()

        sample_rng = cube_selection.GetSize()
        sample_vol = np.prod(sample_rng)

        return sample_mass / sample_vol

    def _calc_contact_no(self, particles, time_step, sample_frac=0.9):
        cube_selection = self._get_cropped_region(particles, time_step, sample_frac)

        all_contacts_x = cube_selection.GetGridFunction("Contact : X").GetArray(
            time_step=time_step
        )
        all_contacts_y = cube_selection.GetGridFunction("Contact : Y").GetArray(
            time_step=time_step
        )
        all_contacts_z = cube_selection.GetGridFunction("Contact : Z").GetArray(
            time_step=time_step
        )

        x_rng, y_rng, z_rng = cube_selection.GetSize()
        x_center, y_center, z_center = cube_selection.GetCenter()

        x_mask = (all_contacts_x >= x_center - x_rng / 2) & (
            all_contacts_x <= x_center + x_rng / 2
        )

        y_mask = (all_contacts_y >= y_center - y_rng / 2) & (
            all_contacts_y <= y_center + y_rng / 2
        )
        z_mask = (all_contacts_z >= z_center - z_rng / 2) & (
            all_contacts_z <= z_center + z_rng / 2
        )

        n_contacts = (
            np.logical_and.reduce((x_mask, y_mask, z_mask)).sum()
            * 2
            / cube_selection.GetNumberOfParticles(time_step=time_step)
        )

        return n_contacts

    def post_process(self, sample_frac=0.9, plot=True):
        time_set = self._study.GetTimeSet()
        timeset_arr = time_set.GetValues()
        try:
            settled_timeset = np.where(
                timeset_arr == (self.params.t_fill + self.params.t_settle)
            )[0][0].item()
        except IndexError:
            raise IndexError(
                "Could not find time step corresponding to end of settling phase."
                "Please ensure that the time step exists and matches the sum of t_fill and t_settle parameters."
                f"Available time steps: {timeset_arr}"
            )

        particles = self._study.GetParticles()

        uncompr_dens = self._calc_bulk_density(
            particles, time_step=settled_timeset, sample_frac=sample_frac
        )
        compr_dens = self._calc_bulk_density(
            particles, time_step=-1, sample_frac=sample_frac
        )

        uncompr_contacts = self._calc_contact_no(
            particles, time_step=settled_timeset, sample_frac=sample_frac
        )
        compr_contacts = self._calc_contact_no(
            particles, time_step=-1, sample_frac=sample_frac
        )
        contacts_ratio = compr_contacts / uncompr_contacts

        n_particles_hist = [
            particles.GetNumberOfParticles(time_step=ts) for ts in time_set
        ]
        n_lost = int(max(n_particles_hist) - n_particles_hist[-1])

        # Handle plotting
        if plot:
            bulk_dens = []
            contacts = []

            for timestep in time_set[1:]:
                bulk_dens_ts = self._calc_bulk_density(
                    particles, timestep, sample_frac=sample_frac
                )
                bulk_dens.append(bulk_dens_ts)

                contact_ts = self._calc_contact_no(
                    particles, timestep, sample_frac=sample_frac
                )
                contacts.append(contact_ts)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(time_set[1:], bulk_dens, label="Bulk Density", color="C0")
            ax.set_xlabel("Time (s)", fontsize=16)
            ax.set_ylabel("Bulk Density (kg/m^3)", fontsize=16)

            ax1 = ax.twinx()
            ax1.plot(time_set[1:], contacts, color="C1", label="Average Contacts")
            ax1.set_ylabel("Average Number of Contacts", fontsize=16)

            ax.grid(visible=True)
            ax1.grid(visible=True)
            fig.legend()
            fig.tight_layout()

            fig.savefig(
                pathlib.Path(str(self.params.plots_dir)) / "ts_bulkdens_contacts.png",
                dpi=300,
            )

        # Write all data

        params_dict = asdict(self.params)

        col_names = list(params_dict.keys())
        col_vals = list(params_dict.values())

        col_names.extend(
            [
                "uncompressed_density",
                "compressed_density",
                "uncompressed_contacts",
                "compressed_contacts",
                "contacts_ratio",
            ]
        )

        col_vals.extend(
            [
                uncompr_dens,
                compr_dens,
                uncompr_contacts,
                compr_contacts,
                contacts_ratio,
            ]
        )

        output_path = pathlib.Path(self.params.project_dir) / "results.csv"
        if not output_path.exists():
            with open(output_path, "w") as f:
                f.write(",".join(col_names) + "\n")
        with open(output_path, "a") as f:
            f.write(",".join(map(str, col_vals)) + "\n")

    def execute(self):
        self.load_meshes(insert=self.insertion)
        self.load_material_properties()
        self.load_interactions()
        self.gen_particle()
        self.sim_physics()
        self.insertion_settings(insert=self.insertion)
        self.move_top_wall(insert=self.insertion)
        self.set_domain_settings()
        self.load_modules()
        self.simulate(insert=self.insertion)
        self.post_process(sample_frac=0.9, plot=True)
