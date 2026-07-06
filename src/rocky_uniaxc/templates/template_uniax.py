#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Jinja2 template for Rocky pre/post-processing uniaxial compression scripts.

This file is rendered by the ``rocky_prepost`` backend at sweep-launch time.
Jinja2 placeholders (e.g. ``{{ variable }}``) are substituted with per-case parameter
values before the script is written to the case directory and executed by
Rocky.
"""

import sys
import os
import pathlib
import sqlite3
from typing import Optional
import importlib.util

import numpy as np
import matplotlib.pyplot as plt


# Import particle shapes using importlib
shapes_spec = importlib.util.spec_from_file_location(
    "particles_shapes",
    "{{SHAPES_MODULE_PATH}}",
)
if not shapes_spec:
    raise ImportError("Could not find the particles_shapes.py file.")
particle_shapes = importlib.util.module_from_spec(shapes_spec)
sys.modules["particles_shapes"] = particle_shapes  # Add to sys.modules
shapes_spec.loader.exec_module(particle_shapes)


# Particle properties
P_RADIUS: float | dict = {{RADIUS_P}}  # m
P_DENSITY: float = {{DENSITY_P}}  # kg/m^3
P_YOUNGMOD: float = {{YOUNGMOD_P}}  # Pa
P_POISSON: float = {{POISSON_P}}  # Poisson ratio

ROLLING_MODEL = "{{ROLLING_MODEL}}"  # 'type_1', 'type_3', 'none', 'custom'
assert ROLLING_MODEL in ["type_1", "type_3", "none", "custom"]
ROLLING_FRICTION: float = {{ROLLING_FRICTION}}

# P-P / P-W properties
PP_SURFACE_ENERGY: float = {{SURFACE_ENERGY_PP}}
PP_DYNAMIC_FRICTION: float = {{DYNAMIC_FRICTION_PP}}
PP_STATIC_FRICTION: float = {{STATIC_FRICTION_PP}}
PP_TANGENTIAL_STIFFNESS_RATIO: float = {{TANGENTIAL_STIFFNESS_RATIO_PP}}
PP_COR: float = {{COR_PP}}

PW_SURFACE_ENERGY: float = {{SURFACE_ENERGY_PW}}
PW_DYNAMIC_FRICTION: float = {{DYNAMIC_FRICTION_PW}}
PW_STATIC_FRICTION: float = {{STATIC_FRICTION_PW}}
PW_TANGENTIAL_STIFFNESS_RATIO: float = {{TANGENTIAL_STIFFNESS_RATIO_PW}}

PW_COR: float = {{COR_PW}}

for i, _p in enumerate(
    [
        PP_SURFACE_ENERGY,
        PP_DYNAMIC_FRICTION,
        PP_STATIC_FRICTION,
        PP_TANGENTIAL_STIFFNESS_RATIO,
        PP_COR,
        PW_SURFACE_ENERGY,
        PW_DYNAMIC_FRICTION,
        PW_STATIC_FRICTION,
        PW_TANGENTIAL_STIFFNESS_RATIO,
        PW_COR,
        PP_ROLLING_FRICTION,
        PW_ROLLING_FRICTION,
        P_POISSON,
    ]
):
    if (i in [10, 11]) and (ROLLING_MODEL == "type_3") and (not _p):
        continue

    if (i in [4, 9]) and (_p < 0 or _p > 1):  # CORs
        raise ValueError(
            f"Expected a value between 0 and 1."
            f"Got {_p} for one of the particle properties."
        )
    if (i in [3, 8]) and (_p < 0 or _p > 1):  # Tangential Stiffness Ratio
        raise ValueError(
            f"Expected a value between 0 and 1 for tangential stiffness ratio.Got {_p}."
        )
    if (i == 12) and (_p < 0 or _p > 0.5):  # Poisson
        raise ValueError(
            f"Expected a value between 0 and 0.5 for Poisson's ratio.Got {_p}."
        )
    if (i in [1, 2, 6, 7, 10, 11]) and (_p < 0):  # Frictions
        raise ValueError(f"Expected a non-negative value.Got {_p} for friction.")
    if (i in [0, 5]) and (_p < 0):  # Surface Energy
        raise ValueError(f"Expected a non-negative value.Got {_p} for surface energy.")

# Contact models
NORMAL_FORCE_MODEL = "{{NORMAL_MODEL}}"
assert NORMAL_FORCE_MODEL in [
    "linear_hysteresis",
    "linear_elastic_viscous",
    "damped_hertzian",
    "custom",
]

TANGENTIAL_FORCE_MODEL = "{{TANG_MODEL}}"
assert TANGENTIAL_FORCE_MODEL in [
    "elastic_coulomb",
    "coulomb_limit",
    "mindlin_deresiewicz",
    "custom",
]

ADHESION_MODEL = "{{ADH_MODEL}}"
assert ADHESION_MODEL in ["none", "constant", "linear", "JKR", "custom"]

PARTICLE_BOX_LEN: float = {{L_BOX}}  # m
T_FILL: float = 0.5  # s
T_SETTLE: float = 1  # s

COMPR_PRESSURE: float = {{P_COMPRESS}}  # Pa
T_COMPRESSION: float = 1  # s

INSERT = True  # If False, use volumetric inlet

# Solver settings
NPROCS = os.environ.get("SLURM_CPUS_ON_NODE", 20)
try:
    NPROCS = int(NPROCS)
except ValueError:
    NPROCS = 20

NEIGHBOUR_SEARCH = None
if NEIGHBOUR_SEARCH is not None:
    assert NEIGHBOUR_SEARCH in ["BVH", "RegularGrid", "SparseGrid"]

PROCESSOR: str = {{XPU}}
assert PROCESSOR in ["CPU", "GPU", "MULTI_GPU"]

# Paths
PROJECT_DIR = pathlib.Path.cwd()
MESHDIR = pathlib.Path.cwd().parent / "meshes_{{L_BOX}}"

# Paths to the Rocky executable - for PyRocky implementation
# BB_ROCKY_PATH = '/rds/bear-apps/2023a/EL8-ice/software/ANSYS_Rocky/2024R2.0/bin/Rocky'
# VM_ROCKY_PATH = '/home/rocky-vm/ansys_inc/v242/rocky/bin/Rocky'

# Flag for creating a new project
_run_flag = True
_resume_flag = False

# Tracking created boxes
active_boxes = {}
active_euls = {}


def setup(filename="uniaxial_compression.rocky") -> None:
    """Set up the Rocky project and study for uniaxial compression.

    If the project file already exists it is opened; otherwise a new
    project is created and saved.

    Args:
        filename: Name of the Rocky project file. Defaults to
            ``"uniaxial_compression.rocky"``.
    """

    global project, study

    # Create Rocky file
    rocky_path = PROJECT_DIR / filename
    rocky_path_str = str(rocky_path.resolve())

    if rocky_path.exists():
        project = app.OpenProject(rocky_path_str)
        study = project.GetStudy()
        _run_flag = False
        if study.CanResumeSimulation():
            _run_flag = True
            _resume_flag = True
    else:
        project = app.CreateProject()
        project.SaveProject(rocky_path_str)
        study = project.GetStudy()
        study.SetName("Uniaxial Compression")
        _run_flag = True


def load_meshes(insert=True) -> None:
    """Load the compression walls and optional insert surface into the project.

    Args:
        insert: If ``True``, also import the insert inlet surface.
    """

    if not _run_flag and not _resume_flag:
        return

    global top_wall, bottom_wall, study

    compr_wall1_stl_path = str(MESHDIR / "compressive_wall1.stl")
    compr_wall2_stl_path = str(MESHDIR / "compressive_wall2.stl")

    # Load Top Wall
    top_wall = study.ImportWall(
        compr_wall1_stl_path, import_scale=1.1, convert_yz=True
    )[0]
    top_wall.SetName("Top Wall")
    top_wall.SetBoundaryMass(1e-6)
    top_wall.SetTranslation([-PARTICLE_BOX_LEN / 2, 0, 0])

    top_wall.SetEnableTime(T_SETTLE + 0.5)

    # Load bottom wall with a slight offset
    # to avoid overlap wth periodic boundary
    bottom_wall = study.ImportWall(
        compr_wall2_stl_path, import_scale=1.1, convert_yz=True
    )[0]
    bottom_wall.SetName("Bottom Wall")
    bottom_wall.SetTranslation([PARTICLE_BOX_LEN / 2 + 1e-6, 0, 0])

    if insert:
        insert_stl_path = str(MESHDIR / "insert.stl")
        global insert_inlet
        insert_inlet = study.ImportSurface(
            insert_stl_path, import_scale=1.0, convert_yz=True
        )[0]

        insert_inlet.SetName("Insert Inlet")
        insert_inlet.SetPivotPoint([0, 0, 0])
        insert_inlet.SetOrientationFromAngles(rotation=[0, 0, -90])
        insert_inlet.SetTranslation([0.45 * PARTICLE_BOX_LEN / 2, 0, 0])
        insert_inlet.SetInvertNormal(True)


def load_material_properties():
    """Create particle and wall materials and assign them to the geometry."""
    if not _run_flag and not _resume_flag:
        return

    global study, top_wall, bottom_wall, wall_mat, particle_mat
    material_collection = study.GetMaterialCollection()

    # Create materials for particles and walls
    particle_mat = material_collection.AddSolidMaterial()
    particle_mat.SetName("Particle Material")
    particle_mat.SetDensity(P_DENSITY, "kg/m3")
    particle_mat.SetYoungsModulus(P_YOUNGMOD, "Pa")
    particle_mat.SetPoissonRatio(P_POISSON)
    particle_mat.SetUseBulkDensity(False)

    wall_mat = material_collection.AddSolidMaterial()
    wall_mat.SetName("Wall Material")
    wall_mat.SetDensity(2700, "kg/m3")
    wall_mat.SetYoungsModulus(5e6, "Pa")
    wall_mat.SetPoissonRatio(0.3)
    wall_mat.SetUseBulkDensity(False)

    # Set the material for the meshes
    top_wall.SetMaterial(wall_mat)
    bottom_wall.SetMaterial(wall_mat)


def load_interactions() -> None:
    if not _run_flag and not _resume_flag:
        return

    global study, particle_mat, wall_mat

    interaction_collection = study.GetMaterialsInteractionCollection()
    pp_interaction = interaction_collection.GetMaterialsInteraction(
        particle_mat, particle_mat
    )
    pw_interaction = interaction_collection.GetMaterialsInteraction(
        particle_mat, wall_mat
    )

    # Set the contact laws for the particle-particle interaction
    pp_interaction.SetRestitutionCoefficient(PP_COR)
    pp_interaction.SetSurfaceEnergy(PP_SURFACE_ENERGY)
    pp_interaction.SetStaticFriction(PP_STATIC_FRICTION)
    pp_interaction.SetDynamicFriction(PP_DYNAMIC_FRICTION)
    pp_interaction.SetTangentialStiffnessRatio(PP_TANGENTIAL_STIFFNESS_RATIO)

    # Set the contact laws for the particle-wall interaction
    pw_interaction.SetRestitutionCoefficient(PW_COR)
    pw_interaction.SetSurfaceEnergy(PW_SURFACE_ENERGY)
    pw_interaction.SetStaticFriction(PW_STATIC_FRICTION)
    pw_interaction.SetDynamicFriction(PW_DYNAMIC_FRICTION)
    pw_interaction.SetTangentialStiffnessRatio(PW_TANGENTIAL_STIFFNESS_RATIO)


def set_psd() -> None:
    """Set the particle size distribution for the particles.

    Supports monodisperse (scalar radius) and polydisperse (dict of radii
    to cumulative percentages) distributions.
    """
    if not _run_flag and not _resume_flag:
        return

    global study, particle_mat, particle, P_RADIUS

    if isinstance(P_RADIUS, float) or isinstance(P_RADIUS, int):
        particle = study.CreateParticle()
        size_distr_lst = particle.GetSizeDistributionList()
        size_distr_lst.Clear()

        psd = size_distr_lst.New()
        psd.SetSize(P_RADIUS, "m")
        psd.SetCumulativePercentage(100)

    # If it is a dictionary, create a particle size distribution
    # with multiple sizes
    elif isinstance(P_RADIUS, dict):
        # Check if the values are valid
        if sum(P_RADIUS.values()) == 1:
            P_RADIUS = {k: v * 100 for k, v in P_RADIUS.items()}
        elif sum(P_RADIUS.values()) == 100:
            pass
        else:
            raise ValueError(
                "The size dict values must sum to 1 or 100."
                "Please provide a valid dictionary."
            )
        # Create a new particle and size distribution list
        particle = study.CreateParticle()
        size_distr_lst = particle.GetSizeDistributionList()
        size_distr_lst.Clear()

        # Create a new PSD for each particle size
        init_pct = 100
        sorted_dict = dict(sorted(P_RADIUS.items(), reverse=True))
        for i, (size, proportion) in enumerate(sorted_dict):
            # Create a new PSD for each particle size
            # and set the size and cumulative percentage
            # Use exec to create a variable with the name of the PSD
            # This is not recommended, but it works for this case ;)
            exec(f"psd{i} = size_distr_lst.New()")
            exec(f'psd{i}.SetSize(size, "m")')
            exec(f"psd{i}.SetCumulativePercentage(init_pct)")
            init_pct -= proportion

    # Set particle material
    particle.SetMaterial(particle_mat)
    if PP_ROLLING_FRICTION != "none":
        particle.SetRollingResistance(PP_ROLLING_FRICTION)

    particle.SetMaterial(particle_mat)
    if PW_ROLLING_FRICTION != "none":
        particle.SetRollingResistance(PW_ROLLING_FRICTION)


def gen_particle(shape_dict: dict[str, float | str]) -> None:
    """Create a particle of a specific shape in the Rocky study.

    Args:
        shape_dict: Dictionary with keys ``"name"``, ``"vert_ar"``,
            ``"horiz_ar"``, ``"n_corners"``, ``"sq_degree"``,
            ``"particle_path"``, and ``"smoothness"``.

    Raises:
        ValueError: If the shape type is unknown or a custom polyhedron
            STL path is invalid.
    """
    global particle, study
    study = app.GetStudy()
    particle = study.CreateParticle()
    shape = shape_dict.get("name")

    # Use shape objects from particles_shapes module
    match shape:
        case "sphere":
            shape_obj = particle_shapes.Sphere(radius=P_RADIUS)
        case "sphero_cylinder":
            vert_ar = shape_dict.get("vert_ar", 1.0)
            shape_obj = particle_shapes.SpheroCylinder(radius=P_RADIUS, vert_ar=vert_ar)
        case "polyhedron":
            vert_ar = shape_dict.get("vert_ar", 1.0)
            horiz_ar = shape_dict.get("horiz_ar", 1.0)
            n_corners = shape_dict.get("n_corners", 6)
            sq_degree = shape_dict.get("sq_degree", 1.0)

            shape_obj = particle_shapes.Polyhedron(
                radius=P_RADIUS,
                vert_ar=vert_ar,
                horiz_ar=horiz_ar,
                n_corners=n_corners,
                superquadric_degree=sq_degree,
            )
        case "custom_polyhedron":
            stl_path = str(shape_dict.get("particle_path", ""))
            if not stl_path or not pathlib.Path(stl_path).resolve().is_file():
                raise ValueError("Custom polyhedron requires a valid STL file path.")

            shape_obj = particle_shapes.CustomPolyhedron(
                stl_path=stl_path, radius=P_RADIUS
            )
        case _:
            raise ValueError(
                f"Unknown shape type: {shape}. "
                "Supported shapes are: 'sphere', 'spherocylinder', 'polyhedron', 'custom_polyhedron'."
            )

    # Instantiate the shape for the particle
    shape_obj.particle2rocky(
        particle=particle, material=particle_mat, fric_rolling=ROLLING_FRICTION
    )


def sim_physics() -> None:
    """Configure contact force models and gravity for the simulation."""

    if not _run_flag and not _resume_flag:
        return

    # Contact models
    physics = study.GetPhysics()
    physics.SetNormalForceModel(NORMAL_FORCE_MODEL)
    physics.SetTangentialForceModel(TANGENTIAL_FORCE_MODEL)
    physics.SetAdhesionModel(ADHESION_MODEL)

    # Gravity
    physics.SetGravityXDirection(-9.81)  # X-positive is upwards (don't ask)
    physics.SetGravityYDirection(0)
    physics.SetGravityZDirection(0)


def insertion_settings(insert=True) -> None:
    """Configure particle insertion settings.

    Args:
        insert: If ``True``, use surface inlet insertion. If ``False``,
            use volumetric insertion.
    """
    if not _run_flag and not _resume_flag:
        return

    fill_box_vol = PARTICLE_BOX_LEN**3  # m^3
    if isinstance(P_RADIUS, dict):
        particle_vol = sum(
            (4 / 3) * np.pi * r**3 * p for r, p in P_RADIUS.items()
        ) / sum(P_RADIUS.values())
    else:
        particle_vol = (4 / 3) * np.pi * P_RADIUS**3  # m^3
    # 0.6 is an avg packing fraction of spherical particles
    # Use less particles to account for non-spherical shapes
    n_particles = np.rint(fill_box_vol * 0.5 / particle_vol).astype(int).item()
    mass_particles = particle_vol * P_DENSITY * n_particles

    if insert:
        flowr = mass_particles / T_FILL  # kg/s

        particle_inlet = study.CreateParticleInlet(insert_inlet, particle)

        input_property_lst = particle_inlet.GetInputPropertiesList()
        input_property_lst[0].SetMassFlowRate(flowr, "kg/s")

        particle_inlet.SetStartTime(0)
        particle_inlet.SetStopTime(T_FILL)
        particle_inlet.DisablePeriodic()
    else:
        study.CreateVolumetricInlet(
            particle=particle,
            name="Volumetric Inlet",
            mass=mass_particles,
            seed_coordinates=[0, 0, 0],  # Center of domain is origin
            use_geometries_to_compute=False,
            box_center=[0, 0, 0],
            box_dimensions=[PARTICLE_BOX_LEN, PARTICLE_BOX_LEN, PARTICLE_BOX_LEN],
        )


def move_top_wall():
    frame_source = study.GetMotionFrameSource()
    top_wall_frame = frame_source.NewFrame()

    motions = top_wall_frame.GetMotions()

    # Drop weightless wall
    drop_wall_motion = motions.New()
    drop_wall_motion.SetType("Free Body Translation")
    free_body = drop_wall_motion.GetTypeObject()
    free_body.SetFreeMotionDirection("x")
    drop_wall_motion.SetStartTime(T_FILL + T_SETTLE)

    # Start compression
    # Account for wall mass
    pressure = 1e-6 * 9.81 - COMPR_PRESSURE * PARTICLE_BOX_LEN**2  # N
    compr_motion = motions.New()
    compr_motion.SetType("Additional Force")
    add_force = compr_motion.GetTypeObject()
    add_force.SetForceValue([pressure, 0, 0], "N")
    compr_motion.SetStartTime(T_FILL + T_SETTLE + 0.1)
    compr_motion.SetStopTime(T_FILL + T_SETTLE + T_COMPRESSION)

    top_wall_frame.ApplyTo(top_wall)


def set_domain_settings() -> None:
    if not _run_flag and not _resume_flag:
        return

    global study

    domain_settings = study.GetDomainSettings()

    # Disable as it is unreliable
    domain_settings.DisableUseBoundaryLimits()
    domain_settings.DisablePeriodicAtGeometryLimits()

    # Adding 1.5 safety factor to avoid issues with particles
    domain_settings.SetDomainType("CARTESIAN")
    domain_settings.SetCoordinateLimitsMinValues(
        [
            (-PARTICLE_BOX_LEN / 2) * 1.5,
            (-PARTICLE_BOX_LEN / 2) * 1.5,
            (-PARTICLE_BOX_LEN / 2) * 1.5,
        ]
    )
    domain_settings.SetCoordinateLimitsMaxValues(
        [
            (PARTICLE_BOX_LEN / 2) * 1.5,
            (PARTICLE_BOX_LEN / 2) * 1.5,
            (PARTICLE_BOX_LEN / 2) * 1.5,
        ]
    )

    # Set the periodic limits for the domain
    # X direction does not matter as walls are there
    domain_settings.SetCartesianPeriodicDirections("YZ")
    domain_settings.SetPeriodicLimitsMinCoordinates(
        [-1e6, -PARTICLE_BOX_LEN / 2, -PARTICLE_BOX_LEN / 2]
    )
    domain_settings.SetPeriodicLimitsMaxCoordinates(
        [1e6, PARTICLE_BOX_LEN / 2, PARTICLE_BOX_LEN / 2]
    )


def _select_processor(solver, processor: str) -> None:
    """Select the simulation processor (CPU or GPU).

    Falls back to CPU with a warning file if the requested GPU is
    unavailable.

    Args:
        solver: The solver object from the Rocky study.
        processor: Processor to use — ``"GPU"`` or ``"CPU"``.
    """
    if processor == "GPU":
        if processor not in solver.GetValidSimulationTargetValues():
            warning_path = os.path.join(PROJECT_DIR, "warnings.txt")
            write_mode = "w" if os.path.exists(warning_path) else "a"
            with open(warning_path, write_mode) as f:
                f.write("GPU was not available - switching to CPU")
            solver.SetSimulationTarget("CPU")
        else:
            solver.SetSimulationTarget("GPU")

    elif processor == "CPU":
        solver.SetSimulationTarget("CPU")
        solver.SetNumberOfProcessors(NPROCS)


def simulate(insert: bool = True, autotimestep: bool = True, timestep=None) -> None:
    """Start the uniaxial compression simulation.

    Args:
        insert: Whether the fill phase is included in timing.
        autotimestep: If ``True``, Rocky determines the timestep
            automatically. Defaults to ``True``.
        timestep: Fixed timestep in seconds. Only used when
            ``autotimestep=False``.
    """

    if not _run_flag:
        return

    global study

    study = project.GetStudy()
    solver = study.GetSolver()
    _select_processor(solver=solver, processor=PROCESSOR)

    if not autotimestep:
        if not timestep:
            solver.SetUseFixedTimestep(True)
            solver.SetFixedTimestep(1e-6, "s")
        else:
            solver.SetUseFixedTimestep(True)
            solver.SetFixedTimestep(timestep, "s")

    if insert:
        runtime = T_FILL + T_SETTLE + T_COMPRESSION
    else:
        runtime = T_SETTLE + T_COMPRESSION
    solver.SetSimulationDuration(runtime, "s")

    project.SaveProject()
    print(f"Running simulation with {PROCESSOR} solver.")
    study.StartSimulation()

    while study.IsSimulating():
        study.RefreshResults()
        print(f"Simulation Progress: {study.GetProgress():.2f} %")
    print("Simulation completed.")


def load_modules():
    """Enable contacts data collection and adhesive contact reporting."""

    global study

    contacts_data = study.GetContactData()
    contacts_data.EnableCollectContactsData()
    if ADHESION_MODEL != "none":
        contacts_data.EnableIncludeAdhesiveContacts()


def _get_cropped_region(particles, time_step, sample_frac=0.9):
    if time_step in active_boxes:
        return active_boxes[time_step]

    x_coords = particles.GetGridFunction("Coordinate : X").GetArray(time_step=time_step)
    y_coords = particles.GetGridFunction("Coordinate : Y").GetArray(time_step=time_step)
    z_coords = particles.GetGridFunction("Coordinate : Z").GetArray(time_step=time_step)

    positions = np.vstack((x_coords, y_coords, z_coords))
    pos_rngs = np.ptp(positions, axis=1)
    sample_rng = pos_rngs * sample_frac
    processes = project.GetUserProcessCollection()
    cube_selection = processes.CreateCubeProcess(particles)
    cube_selection.SetCenter(x_coords.mean(), y_coords.mean(), z_coords.mean())
    cube_selection.SetSize(sample_rng[0], sample_rng[1], sample_rng[2])

    active_boxes[time_step] = cube_selection
    return cube_selection


def _calc_bulk_dens(particles, time_step, sample_frac=0.9) -> float:
    """Calculate bulk density at a given time step.

    Samples a fraction of the domain and computes
    :math:`\\rho = m / V`.

    Args:
        particles: Rocky particles collection.
        time_step: Time-step index.
        sample_frac: Fraction of the domain to sample. Defaults to 0.9.

    Returns:
        Bulk density in kg/m³.
    """
    cube_selection = _get_cropped_region(particles, time_step, sample_frac)

    mass_arr = cube_selection.GetGridFunction("Particle Mass").GetArray(
        time_step=time_step
    )
    sample_mass = mass_arr.sum()

    sample_rng = cube_selection.GetSize()
    sample_vol = np.prod(sample_rng)

    return sample_mass / sample_vol


def _calc_contact_no(particles, time_step, sample_frac: float = 0.9) -> float:
    cube_selection = _get_cropped_region(particles, time_step, sample_frac=sample_frac)
    contact_data = study.GetContactData()

    all_contacts_x = contact_data.GetGridFunction("Contact : Coordinate : X").GetArray(
        time_step=time_step
    )
    all_contacts_y = contact_data.GetGridFunction("Contact : Coordinate : Y").GetArray(
        time_step=time_step
    )
    all_contacts_z = contact_data.GetGridFunction("Contact : Coordinate : Z").GetArray(
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


def _calc_shear_strength(particles, time_step, sample_frac=0.9) -> float:
    cube_selection = _get_cropped_region(particles, time_step, sample_frac)

    if time_step in active_euls.keys():
        eul_region = active_euls[time_step]
    else:
        processes = project.GetUserProcessCollection()
        eul_region = processes.CreateEulerianStatistics(cube_selection)
        eul_region.SetDivisions([1, 1, 1])
        active_euls[time_step] = eul_region

    stress_mat = []
    for comp in [["XX", "XY", "XZ"], ["XY", "YY", "YZ"], ["XZ", "YZ", "ZZ"]]:
        stress_mat.append(
            [
                eul_region.GetGridFunction(f"Stress Component {comp[0]}").GetArray(
                    time_step=time_step
                )[0],
                eul_region.GetGridFunction(f"Stress Component {comp[1]}").GetArray(
                    time_step=time_step
                )[0],
                eul_region.GetGridFunction(f"Stress Component {comp[2]}").GetArray(
                    time_step=time_step
                )[0],
            ]
        )

    stress_mat = np.array(stress_mat)
    evals, _ = np.linalg.eigh(stress_mat)
    idx = evals.argsort()[::-1]
    princ_stresses = evals[idx]

    sig1, sig2, sig3 = princ_stresses
    mean_stress = (sig1 + sig2 + sig3) / 3
    dev_stress = sig1 - sig3

    return mean_stress, dev_stress


def post_process(plot: Optional[bool] = True) -> None:
    """Post-process simulation results.

    Computes uncompressed and compressed bulk densities, Hausner ratio,
    compression index, contact numbers, and shear strengths.  Optionally
    generates plots and writes results to CSV and SQLite.

    Args:
        plot: If ``True``, generate and save time-series plots. Defaults
            to ``True``.
    """
    global study, project, particle

    time_set = study.GetTimeSet()
    timeset_arr = time_set.GetValues()
    target_time = T_FILL + T_SETTLE
    settled_timestep = np.argmin(np.abs(timeset_arr - target_time)).item()
    if abs(timeset_arr[settled_timestep] - target_time) > 1e-3:
        raise IndexError("Matched time step is too far from target time")
    particles = study.GetParticles()

    # Calculate bulk densities
    uncompr_dens = _calc_bulk_dens(particles, settled_timestep, 0.9)
    compr_dens = _calc_bulk_dens(particles, -1, 0.9)
    hausner_ratio = compr_dens / uncompr_dens
    compr_idx = 100 * (compr_dens - uncompr_dens) / compr_dens

    # Calculate contact numbers
    uncompr_contacts = _calc_contact_no(particles, settled_timestep, 0.9)
    compr_contacts = _calc_contact_no(particles, -1, 0.9)
    contacts_ratio = compr_contacts / uncompr_contacts

    # Calculate shear strengths
    uncompr_mean_stress, uncompr_dev_stress = _calc_shear_strength(
        particles, settled_timestep, 0.9
    )
    compr_mean_stress, compr_dev_stress = _calc_shear_strength(particles, -1, 0.9)
    uncompr_stress_ratio = (
        uncompr_dev_stress / uncompr_mean_stress if uncompr_mean_stress != 0 else 0
    )
    compr_stress_ratio = (
        compr_dev_stress / compr_mean_stress if compr_mean_stress != 0 else 0
    )

    bulk_dens = []
    contacts = []
    mean_stresses = []
    dev_stresses = []

    if plot:
        PLOTS_DIR = os.path.join(PROJECT_DIR, "plots")
        for timestep in time_set[1:]:
            bulk_dens_ts = _calc_bulk_dens(particles, timestep, sample_frac=0.9)
            bulk_dens.append(bulk_dens_ts)
            contact_ts = _calc_contact_no(particles, timestep, sample_frac=0.9)
            contacts.append(contact_ts)
            mean_stress_ts, dev_stress_ts = _calc_shear_strength(
                particles, timestep, sample_frac=0.9
            )
            mean_stresses.append(mean_stress_ts)
            dev_stresses.append(dev_stress_ts)

        # Plot bulk dens
        bulk_dens_t = np.array(bulk_dens)
        fig, ax = plt.subplots(figsize=(10, 6))
        color = "tab:blue"
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Bulk Density (kg/m³)", color=color)
        ax.plot(timeset_arr[1:], bulk_dens_t, color=color)
        ax.tick_params(axis="y", labelcolor=color)

        ax.set_title("Bulk Density vs Time")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "bulk_density.png"), dpi=300)
        plt.close(fig)

        # Plot contact no
        fig, ax = plt.subplots(figsize=(10, 6))
        color = "tab:orange"
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Average Contacts per Particle", color=color)
        ax.plot(timeset_arr[1:], contacts, color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax.set_title("Average Contacts per Particle vs Time")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "contact_no.png"), dpi=300)
        plt.close(fig)

        # Plot mean and deviatoric stresses
        mean_stresses_t = np.array(mean_stresses)
        dev_stresses_t = np.array(dev_stresses)
        fig, ax = plt.subplots(figsize=(10, 6))
        color = "tab:green"
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean Stress (Pa)", color=color)
        ax.plot(timeset_arr[1:], mean_stresses_t, color=color, label="Mean Stress")
        ax.tick_params(axis="y", labelcolor=color)
        ax2 = ax.twinx()
        color = "tab:red"
        ax2.set_ylabel("Deviatoric Stress (Pa)", color=color)
        ax2.plot(
            timeset_arr[1:], dev_stresses_t, color=color, label="Deviatoric Stress"
        )
        ax2.tick_params(axis="y", labelcolor=color)
        ax.set_title("Mean and Deviatoric Stresses vs Time")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "stresses.png"), dpi=300)
        plt.close(fig)

    case_num = int(os.path.basename(PROJECT_DIR).split("_")[-1])
    n_particles = int(particles.GetNumberOfParticles(0))
    particles_lost = int(n_particles - particles.GetNumberOfParticles(-1))

    shape_name = particle.GetShape()
    vert_ar = particle.GetVerticalAspectRatio()
    horiz_ar = particle.GetHorizontalAspectRatio()
    n_corners = particle.GetNumberOfCorners()
    sq_degree = particle.GetSuperquadricDegree()
    smoothness = particle.GetSmoothness()

    col_vals = [
        case_num,
        P_RADIUS,
        P_DENSITY,
        P_YOUNGMOD,
        P_POISSON,
        PP_SURFACE_ENERGY,
        PP_DYNAMIC_FRICTION,
        PP_STATIC_FRICTION,
        PP_ROLLING_FRICTION,
        PP_TANGENTIAL_STIFFNESS_RATIO,
        PP_COR,
        PW_SURFACE_ENERGY,
        PW_DYNAMIC_FRICTION,
        PW_STATIC_FRICTION,
        PW_ROLLING_FRICTION,
        PW_TANGENTIAL_STIFFNESS_RATIO,
        PW_COR,
        COMPR_PRESSURE,
        NORMAL_FORCE_MODEL,
        TANGENTIAL_FORCE_MODEL,
        ADHESION_MODEL,
        ROLLING_MODEL,
        PARTICLE_BOX_LEN,
        n_particles,
        shape_name,
        vert_ar,
        horiz_ar,
        n_corners,
        sq_degree,
        smoothness,
        uncompr_dens,
        compr_dens,
        hausner_ratio,
        compr_idx,
        particles_lost,
        uncompr_contacts,
        compr_contacts,
        contacts_ratio,
    ]

    col_names = [
        "case_n",
        "p_radius",
        "p_density",
        "p_youngmod",
        "p_poisson",
        "pp_surface_energy",
        "pp_dynamic_friction",
        "pp_static_friction",
        "pp_rolling_friction",
        "pp_tangential_stiffness_ratio",
        "pp_cor",
        "pw_surface_energy",
        "pw_dynamic_friction",
        "pw_static_friction",
        "pw_rolling_friction",
        "pw_tangential_stiffness_ratio",
        "pw_cor",
        "compression_pressure",
        "normal_force_model",
        "tangential_force_model",
        "adhesion_model",
        "rolling_model",
        "box_len",
        "n_particles",
        "shape_name",
        "vert_ar",
        "horiz_ar",
        "n_corners",
        "sq_degree",
        "smoothness",
        "bulk_density",
        "compressed_density",
        "hausner_ratio",
        "compression_index",
        "n_lost",
        "uncompr_contacts",
        "compr_contacts",
        "contacts_ratio",
    ]

    assert len(col_vals) == len(col_names), (
        "Column values and names must have the same length."
    )

    # Write results to a CSV file
    with open("results.csv", "w") as f:
        # Convert col_names to a comma-separated string
        f.write(",".join(col_names))
        f.write("\n")
        # Convert cols to a comma-separated string
        f.write(",".join(map(str, col_vals)))

    # Dump the results to a SQLite database
    sqlite_path = os.path.abspath("../results.db")
    with sqlite3.connect(sqlite_path) as conn:
        cursor = conn.cursor()

        create_table_query = """CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_n INTEGER,
            p_radius REAL,
            p_density REAL,
            p_youngmod REAL,
            p_poisson REAL,
            pp_surface_energy REAL,
            pp_dynamic_friction REAL,
            pp_static_friction REAL,
            pp_rolling_friction REAL,
            pp_tangential_stiffness_ratio REAL,
            pp_cor REAL,
            pw_surface_energy REAL,
            pw_dynamic_friction REAL,
            pw_static_friction REAL,
            pw_rolling_friction REAL,
            pw_tangential_stiffness_ratio REAL,
            pw_cor REAL,
            compression_pressure REAL,
            normal_force_model TEXT,
            tangential_force_model TEXT,
            adhesion_model TEXT,
            rolling_model TEXT,
            box_len REAL,
            n_particles INTEGER,
            shape_name TEXT,
            vert_ar REAL,
            horiz_ar REAL,
            n_corners INTEGER,
            sq_degree REAL,
            smoothness REAL,
            bulk_density REAL,
            compressed_density REAL,
            hausner_ratio REAL,
            compression_index REAL,
            n_lost INTEGER,
            n_uncompr_contacts REAL,
            n_compr_contacts REAL,
            contacts_ratio REAL
        )"""
        insert_query = f"""INSERT INTO results (
            case_n, p_radius, p_density, p_youngmod, p_poisson,
            pp_surface_energy, pp_dynamic_friction, pp_static_friction, pp_rolling_friction, pp_tangential_stiffness_ratio, pp_cor,
            pw_surface_energy, pw_dynamic_friction, pw_static_friction, pw_rolling_friction, pw_tangential_stiffness_ratio, pw_cor,
            compression_pressure,
            normal_force_model, tangential_force_model, adhesion_model,
            rolling_model, box_len, n_particles, shape_name, vert_ar, horiz_ar, n_corners,
            sq_degree, smoothness, bulk_density, compressed_density,
            hausner_ratio, compression_index, n_lost, n_uncompr_contacts,
            n_compr_contacts, contacts_ratio
        ) VALUES ({",".join(["?"] * len(col_vals))})
        """

        try:
            cursor.execute(create_table_query)
            cursor.execute(insert_query, col_vals)
            conn.commit()

        except sqlite3.Error as e:
            raise RuntimeError(f"SQLite error: {e}")

        finally:
            project.SaveProject()
            project.CloseProject(check_save_state=False)


shape_dict = {
    "name": "{{SHAPE}}",
    "vert_ar": {{VERT_AR}},
    "horiz_ar": {{HORIZ_AR}},
    "n_corners": {{N_CORNERS}},
    "sq_degree": {{SQ_DEGREE}},
    "particle_path": "{{PARTICLE_PATH}}",
    "smoothness": {{SMOOTHNESS}},
}


INSERT = True

setup()
load_meshes(insert=INSERT)
load_material_properties()
load_interactions()
gen_particle(shape_dict)
sim_physics()
insertion_settings(insert=INSERT)
set_domain_settings()
move_top_wall()
load_modules()
simulate(insert=INSERT)
post_process()
