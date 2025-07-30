#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import sqlite3
import warnings
import importlib.util

import numpy as np
import matplotlib.pyplot as plt


# Import particle shapes usingg importlib
shapes_spec = importlib.util.spec_from_file_location(
    'particles_shapes', os.path.abspath('../../particles_shapes.py'))
if not shapes_spec:
    raise ImportError("Could not find the particles_shapes.py file.")
particle_shapes = importlib.util.module_from_spec(shapes_spec)
sys.modules['particles_shapes'] = particle_shapes  # Add to sys.modules
shapes_spec.loader.exec_module(particle_shapes)


# Particle properties
P_RADIUS: float = {{RADIUS_P}}  # m
P_DENSITY: float = {{DENSITY_P}}  # kg/m^3
P_YOUNGMOD: float = {{YOUNGMOD_P}}  # Pa
P_POISSON: float = {{POISSON_P}}  # Poisson ratio

ROLLING_MODEL = '{{ROLLING_MODEL}}'  # 'type_1', 'type_3', 'none', 'custom'
assert ROLLING_MODEL in ['type_1', 'type_3', 'none', 'custom']

# P-P / P-W properties
PP_DYNAMIC_FRICTION: float = {{DYNAMIC_FRICTION_PP}}
PP_STATIC_FRICTION: float = {{STATIC_FRICTION_PP}}
PP_COR: float = {{COR_PP}}
ROLLING_FRICTION: float = {{ROLLING_FRICTION}}

PW_DYNAMIC_FRICTION: float = {{DYNAMIC_FRICTION_PW}}
PW_STATIC_FRICTION: float = {{STATIC_FRICTION_PW}}
PW_COR: float = {{COR_PW}}

for _p in [
    PP_DYNAMIC_FRICTION, PP_STATIC_FRICTION, PP_COR,
    PW_DYNAMIC_FRICTION, PW_STATIC_FRICTION, PW_COR,
    ROLLING_FRICTION, P_POISSON]:
    if _p < 0 or _p > 1:
        raise ValueError(
            f"Expected a value between 0 and 1."
            f"Got {_p} for one of the particle properties."
        )

# Contact models
NORMAL_FORCE_MODEL = '{{NORMAL_MODEL}}'
assert NORMAL_FORCE_MODEL in [
    'linear_hysteresis',
    'linear_elastic_viscous',
    'damped_hertzian',
    'custom']

TANGENTIAL_FORCE_MODEL = '{{TANG_MODEL}}'
assert TANGENTIAL_FORCE_MODEL in [
    'elastic_coulomb',
    'coulomb_limit',
    'mindlin_deresiewicz',
    'custom']

ADHESION_MODEL = '{{ADH_MODEL}}'
assert ADHESION_MODEL in ['none', 'constant', 'linear', 'JKR', 'custom']

PARTICLE_BOX_LEN: float = {{L_BOX}}  # m
T_FILL: float = 0  # s
T_SETTLE: float = 1.5  # s

COMPR_PRESSURE: float = {{P_COMPRESS}}  # Pa
T_COMPRESSION: float = 1  # s

# Insert type
INSERT_TYPE: str = 'vol'  # 'vol', 'ins'
assert INSERT_TYPE in ['vol', 'ins']

# Solver settings
NPROCS: int = os.environ.get('SLURM_CPUS_ON_NODE', 20)
NEIGHBOUR_SEARCH: str = None
if NEIGHBOUR_SEARCH is not None:
    assert NEIGHBOUR_SEARCH in ['BVH', 'RegularGrid', 'SparseGrid']

PROCESSOR: str = 'CPU'
assert PROCESSOR in ['CPU', 'GPU', 'MULTI_GPU']
RUNTIME: float = 5.  # s
assert RUNTIME >= sum([T_FILL, T_SETTLE, T_COMPRESSION])

# Paths
PROJECT_DIR = os.getcwd()
MESHDIR = os.path.abspath(f'../{{MESH_DIR}}_{PARTICLE_BOX_LEN}')

# Paths to the Rocky executable - for PyRocky implementation
# BB_ROCKY_PATH = '/rds/bear-apps/2023a/EL8-ice/software/ANSYS_Rocky/2024R2.0/bin/Rocky'
# VM_ROCKY_PATH = '/home/rocky-vm/ansys_inc/v242/rocky/bin/Rocky'

# Flag for creating a new project
_run_flag = True
_resume_flag = False


def setup(filename='uniaxial_compression.rocky') -> None:
    """
    Setup the Rocky project and study for uniaxial compression simulation.
    If the project file already exists, it will load the existing project.
    Otherwise, it will create a new project and study.
    """

    global project, study

    rocky_path = os.path.join(PROJECT_DIR, filename)
    if os.path.exists(rocky_path):
        project = app.OpenProject(rocky_path)
        study = project.GetStudy()
        _run_flag = False
        if study.CanResumeSimulation():
            _run_flag = True
            _resume_flag = True
    else:
        project = app.CreateProject()
        project.SaveProject(rocky_path)
        study = project.GetStudy()
        study.SetName('Uniaxial Compression')
        _run_flag = True


def load_meshes() -> None:
    """
    Load the walls into the Rocky project.
    """

    if not _run_flag and not _resume_flag:
        return

    global top_wall, bottom_wall, study

    compr_wall1_stl_path = os.path.join(
        MESHDIR, 'compressive_wall1.stl'
    )
    compr_wall2_stl_path = os.path.join(
        MESHDIR, 'compressive_wall2.stl'
    )

    # Load Top Wall
    top_wall = study.ImportWall(
        compr_wall1_stl_path, import_scale=1.0,
        convert_yz=True
    )[0]
    top_wall.SetName('Top Wall')
    top_wall.SetBoundaryMass(1e-6)
    top_wall.SetTranslation(
        [-PARTICLE_BOX_LEN / 2, 0, 0 ]
    )

    # Load bottom wall with a slight offset
    # to avoid overlap wth periodic boundary
    bottom_wall = study.ImportWall(
        compr_wall2_stl_path, import_scale=1.0,
        convert_yz=True
    )[0]
    bottom_wall.SetName('Bottom Wall')
    bottom_wall.SetTranslation(
        [PARTICLE_BOX_LEN / 2 + 1e-6, 0, 0]
    )

    if INSERT_TYPE == 'ins':
        insert_stl_path = os.path.join(
            MESHDIR, 'insert.stl'
        )

        global insert_inlet
        insert_inlet = study.ImportSurface(
            insert_stl_path,
            import_scale=0.99,
            convert_yz=True)[0]
        insert_inlet.SetName('Insert Inlet')


def load_material_properties():
    """
    Load the material properties for the particles and walls.
    """
    if not _run_flag and not _resume_flag:
        return

    global study, top_wall, bottom_wall, wall_mat, particle_mat
    material_collection = study.GetMaterialCollection()

    particle_mat = material_collection.AddSolidMaterial()
    particle_mat.SetName("Particle Material")
    particle_mat.SetDensity(P_DENSITY, 'kg/m3')
    particle_mat.SetYoungsModulus(P_YOUNGMOD, 'Pa')
    particle_mat.SetPoissonRatio(P_POISSON)
    particle_mat.SetUseBulkDensity(False)

    wall_mat = material_collection.AddSolidMaterial()
    wall_mat.SetName("Wall Material")
    wall_mat.SetDensity(2700, 'kg/m3')
    wall_mat.SetYoungsModulus(5e6, 'Pa')
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
        particle_mat,
        particle_mat
    )
    pw_interaction = interaction_collection.GetMaterialsInteraction(
        particle_mat,
        wall_mat
    )

    # Set the contact laws for the particle-particle interaction
    pp_interaction.SetRestitutionCoefficient(PP_COR)
    pp_interaction.SetStaticFriction(PP_STATIC_FRICTION)
    pp_interaction.SetDynamicFriction(PP_DYNAMIC_FRICTION)

    # Set the contact laws for the particle-wall interaction
    pw_interaction.SetRestitutionCoefficient(PW_COR)
    pw_interaction.SetStaticFriction(PW_STATIC_FRICTION)
    pw_interaction.SetDynamicFriction(PW_DYNAMIC_FRICTION)


def set_psd() -> None:
    """
    Set the particle size distribution for the particles.
    """
    if not _run_flag and not _resume_flag:
        return

    global study, particle_mat, particle, P_RADIUS

    if isinstance(P_RADIUS, float) or isinstance(P_RADIUS, int):
        particle = study.CreateParticle()
        size_distr_lst = particle.GetSizeDistributionList()
        size_distr_lst.Clear()

        psd = size_distr_lst.New()
        psd.SetSize(P_RADIUS, 'm')
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
            exec(f'psd{i} = size_distr_lst.New()')
            exec(f'psd{i}.SetSize(size, "m")')
            exec(f'psd{i}.SetCumulativePercentage(init_pct)')
            init_pct -= proportion

    # Set particle material
    particle.SetMaterial(particle_mat)
    if ROLLING_FRICTION != 'none':
        particle.SetRollingResistance(ROLLING_FRICTION)

def gen_particle(shape_dict: dict[str, float|str]) -> None:
    """
    Create a particle of a specific shape.
    """
    global particle

    particle = study.CreateParticle()
    shape = shape_dict.get("name")
    match shape:
        case "sphere":
            shape_obj = particle_shapes.Sphere(radius=P_RADIUS)
        case "spherocylinder":
            vert_ar = shape_dict.get("vert_ar", 1.0)
            shape_obj = particle_shapes.SpheroCylinder(radius=P_RADIUS, vert_ar=vert_ar)
        case "polyhedron":
            vert_ar = shape_dict.get("vert_ar", 1.0)
            horiz_ar = shape_dict.get("horiz_ar", 1.0)
            n_corners = shape_dict.get("n_corners", 6)
            sq_degree = shape_dict.get("sq_degree", 1.0)

            shape_obj = particle_shapes.Polyhedron(
                radius=P_RADIUS, vert_ar=vert_ar,
                horiz_ar=horiz_ar, n_corners=n_corners, 
                superquadric_degree=sq_degree
            )
        case "custom_polyhedron":
            stl_path = str(shape_dict.get("particle_path", ""))
            if not stl_path or not os.path.exists(stl_path):
                raise ValueError(
                    "Custom polyhedron requires a valid STL file path."
                )

            shape_obj = particle_shapes.CustomPolyhedron(
                stl_path=stl_path,
                radius=P_RADIUS
            )
        case _:
            raise ValueError(
                f"Unknown shape type: {shape}. "
                "Supported shapes are: 'sphere', 'spherocylinder', 'polyhedron', 'custom_polyhedron'."
            )
        
    # Instantiate the shape for the particle
    shape_obj.instantiate_shape(particle=particle)

    if ROLLING_MODEL != 'none':
        shape_obj.set_psd(material=particle_mat, rolling_friction=ROLLING_FRICTION)
    else:
        shape_obj.set_psd(material=particle_mat)

def sim_physics() -> None:
    """
    Set the physics for the simulation.
    """

    if not _run_flag and not _resume_flag:
        return

    physics = study.GetPhysics()
    physics.SetNormalForceModel(NORMAL_FORCE_MODEL)
    physics.SetTangentialForceModel(TANGENTIAL_FORCE_MODEL)
    physics.SetAdhesionModel(ADHESION_MODEL)

    physics.SetGravityXDirection(-9.81)
    physics.SetGravityYDirection(0)
    physics.SetGravityZDirection(0)


def insertion_settings() -> None:
    """
    Set the insertion settings for the particles.
    """
    if not _run_flag and not _resume_flag:
        return

    fill_box_vol = PARTICLE_BOX_LEN**3  # m^3
    particle_vol = (4 / 3) * np.pi * P_RADIUS**3  # m^3

    # 0.64 is an avg packing fraction of sphereical particles
    if INSERT_TYPE == 'ins':
        n_particles = np.rint(
            fill_box_vol * 0.64 / particle_vol
        ).astype(int).item()

        mass_particles = particle_vol * P_DENSITY * n_particles
        flowr = mass_particles / T_FILL  # kg/s

        particle_inlet = study.CreateParticleInlet(
            insert_inlet, particle
        )

        input_property_lst = particle_inlet.GetInputPropertiesList()
        input_property_lst[0].SetMassFlowRate(flowr, 'kg/s')

        particle_inlet.SetStartTime(0)
        particle_inlet.SetStopTime(T_FILL)
        particle_inlet.DisablePeriodic()
    else:
        fill_box_vol = PARTICLE_BOX_LEN**3  # m^3
        particle_vol = (4 / 3) * np.pi * P_RADIUS**3  # m^3
        n_particles = fill_box_vol / particle_vol
        mass_particles = particle_vol * P_DENSITY * n_particles

        study.CreateVolumetricInlet(
            particle=particle,
            name='Volumetric Inlet',
            mass=mass_particles,
            seed_coordinates=[0, 0, 0],
            use_geometries_to_compute=False,
            box_center=[0, 0, 0],
            box_dimensions=[PARTICLE_BOX_LEN, PARTICLE_BOX_LEN, PARTICLE_BOX_LEN]
        )


def set_domain_settings() -> None:

    if not _run_flag and not _resume_flag:
        return

    global study

    # Domain settings for the simulation
    domain_settings = study.GetDomainSettings()
    domain_settings.SetDomainType('CARTESIAN')
    domain_settings.SetCoordinateLimitsMinValues(
        [-PARTICLE_BOX_LEN / 2 - 1e6, -PARTICLE_BOX_LEN / 2, -PARTICLE_BOX_LEN / 2]
    )
    domain_settings.SetCoordinateLimitsMaxValues(
        [PARTICLE_BOX_LEN / 2 + 1e6, PARTICLE_BOX_LEN / 2, PARTICLE_BOX_LEN / 2]
    )

    # Set the periodic limits for the domain
    domain_settings.SetCartesianPeriodicDirections('YZ')
    domain_settings.SetPeriodicLimitsMinCoordinates(
        [-PARTICLE_BOX_LEN / 2 - 1e6, -PARTICLE_BOX_LEN / 2, -PARTICLE_BOX_LEN / 2]
    )

    domain_settings.SetPeriodicLimitsMaxCoordinates(
        [PARTICLE_BOX_LEN / 2 + 1e6, PARTICLE_BOX_LEN / 2, PARTICLE_BOX_LEN / 2]
    )


def simulate(autotimestep: bool=True, timestep=None) -> None:

    if not _run_flag:
        return

    global study

    study = project.GetStudy()
    solver = study.GetSolver()
    solver.SetNumberOfProcessors(int(NPROCS))

    if not autotimestep:
        if not timestep:
            solver.SetUseFixedTimestep(True)
            solver.SetFixedTimestep(1e-6, 's')
        else:
            solver.SetUseFixedTimestep(True)
            solver.SetFixedTimestep(timestep, 's')

    solver.SetSimulationDuration(RUNTIME, 's')

    if INSERT_TYPE == 'ins':
        solver.SetReleaseParticlesWithoutOverlapCheck(True)

    project.SaveProject()
    print(f"Running simulation with {PROCESSOR} solver.")
    study.StartSimulation()

    while study.IsSimulating():
        study.RefreshResults()
        print(f"Simulation Progress: {study.GetProgress():.2f} %")
    print("Simulation completed.")

def settle_particles(autotimestep: bool=True, timestep=None):

    global study, project

    if not _run_flag or study.HasResults():
        return


    if not autotimestep:
        if not timestep:
            solver.SetUseFixedTimestep(True)
            solver.SetFixedTimestep(1e-6, 's')
        else:
            solver.SetUseFixedTimestep(True)
            solver.SetFixedTimestep(timestep, 's')

    solver = study.GetSolver()
    solver.SetSimulationDuration(T_SETTLE, 's')
    study.StartSimulation()
    while study.IsSimulating():
        study.RefreshResults()
        print(f"Simulation Progress: {study.GetProgress():.2f} %")

    # Find current wall position
    old_wall_pos = top_wall.GetGridFunction('Coordinate : Nodal : X')\
        .GetArray(time_step=-1).max()

    # Calculate new wall position -> 1 micron above bed height
    particles = study.GetParticles()

    # Handle P_RADIUS being either a float or a dictionary
    max_radius = P_RADIUS if isinstance(P_RADIUS, float) or isinstance(P_RADIUS, int) else max(P_RADIUS.keys())
    new_wall_pos = particles.GetGridFunction(
        'Coordinate : X').GetArray(time_step=-1).max()\
        + max_radius + 1e-6

    global wall_pos
    wall_pos = (old_wall_pos, new_wall_pos)

    project.SaveProjectForRestart('uniaxial_compression_restart.rocky')

def compress_particles(autotimestep: bool=True, timestep=None):

    global study, project, top_wall, wall_pos
    if not _run_flag:
        raise RuntimeError(
            "Simulation has not been run yet. "
            "Please run the simulation before compressing particles."
        )
    old_wall_pos, new_wall_pos = wall_pos

    frame_source = study.GetMotionFrameSource()

    frame_source = study.GetMotionFrameSource()
    compr_motion_frame = frame_source.NewFrame()
    motions = compr_motion_frame.GetMotions()

    # If the wall is already at the new position, skip the motion
    if old_wall_pos != new_wall_pos:
        lower_wall_motion = motions.New()
        lower_wall_motion.SetType('Translation')
        translation = lower_wall_motion.GetTypeObject()
        translation.SetInput('fixed_velocity')
        translation.SetVelocity(
            [new_wall_pos-old_wall_pos, 0, 0], 'm/s')
        lower_wall_motion.SetStartTime(0)
        lower_wall_motion.SetStopTime(0.5)

    # Handling the free body motion
    freebody_motion = motions.New()
    freebody_motion.SetType('Free Body Translation')
    freebody = freebody_motion.GetTypeObject()
    freebody.SetFreeMotionDirection('x')
    freebody_motion.SetStartTime(0.5)

    # Set the compression wall motion
    force_magnitude = COMPR_PRESSURE * PARTICLE_BOX_LEN**2
    force_motion = motions.New()
    force_motion.SetType('Additional Force')
    add_force = force_motion.GetTypeObject()
    add_force.SetForceValue([-force_magnitude, 0, 0], 'N')
    force_motion.SetStartTime(0.5)

    compr_motion_frame.ApplyTo(top_wall)
    solver = study.GetSolver()
    solver.SetSimulationDuration(T_COMPRESSION, 's')

    if not autotimestep:
        if not timestep:
            solver.SetUseFixedTimestep(True)
            solver.SetFixedTimestep(1e-6, 's')
        else:
            solver.SetUseFixedTimestep(True)
            solver.SetFixedTimestep(timestep, 's')

    study.StartSimulation()
    project.SaveProject()

def _calc_bulk_dens(particles, time_step, sample_frac=0.8) -> float:
    x_coords = particles.GetGridFunction(
        'Coordinate : X').GetArray(time_step=time_step)
    y_coords = particles.GetGridFunction(
        'Coordinate : Y').GetArray(time_step=time_step)
    z_coords = particles.GetGridFunction(
        'Coordinate : Z').GetArray(time_step=time_step)

    positions = np.vstack((x_coords, y_coords, z_coords))
    pos_rngs = np.ptp(positions, axis=1)
    sample_rng = pos_rngs * sample_frac

    processes = project.GetUserProcessCollection()
    cube_selection = processes.CreateCubeProcess(particles)
    cube_selection.SetCenter(
        x_coords.mean(),
        y_coords.mean(),
        z_coords.mean()
    )
    cube_selection.SetSize(
        sample_rng[0],
        sample_rng[1],
        sample_rng[2]
    )

    mass_arr = cube_selection.GetGridFunction(
        'Particle Mass').GetArray(time_step=time_step)
    sample_mass = mass_arr.sum()
    sample_vol = sample_rng.prod()

    return sample_mass / sample_vol


def _calc_bulk_dens_v2(particles, time_step, sample_frac=0.8    ) -> tuple:

    # My clever way of importing the packing3d module
    import importlib.util
    packing_path = os.path.abspath('../../packing3d')
    init_file = os.path.join(packing_path, '__init__.py')

    spec = importlib.util.spec_from_file_location("packing3d", init_file)
    packing3d = importlib.util.module_from_spec(spec)
    sys.modules["packing3d"] = packing3d  # Add to sys.modules
    spec.loader.exec_module(packing3d)

    x_coords = particles.GetGridFunction(
        'Coordinate : X').GetArray(time_step=time_step)
    y_coords = particles.GetGridFunction(
        'Coordinate : Y').GetArray(time_step=time_step)
    z_coords = particles.GetGridFunction(
        'Coordinate : Z').GetArray(time_step=time_step)
    radii = particles.GetGridFunction(
        'Particle Size').GetArray(time_step=time_step) / 2.

    positions = np.vstack((x_coords, y_coords, z_coords))
    pos_rngs = np.ptp(positions, axis=1)
    sample_rng = pos_rngs * sample_frac

    boundaries = {
        "x_min": x_coords.mean() - sample_rng[0] / 2,
        "x_max": x_coords.mean() + sample_rng[0] / 2,
        "y_min": y_coords.mean() - sample_rng[1] / 2,
        "y_max": y_coords.mean() + sample_rng[1] / 2,
        "z_min": z_coords.mean() - sample_rng[2] / 2,
        "z_max": z_coords.mean() + sample_rng[2] / 2
    }
    
    # Using @fjbarter's clever code to compute packing fraction
    packing_frac = packing3d.compute_packing_cartesian(
        x_data=x_coords, y_data=y_coords,
        z_data=z_coords, radii=radii,
        boundaries=boundaries
    )

    bulk_dens = packing_frac * P_DENSITY
    voidage = 1 - packing_frac

    return bulk_dens, voidage


def post_process(plot: bool = True, bulk_dens_method='precise') -> None:
    """
    Post-process the simulation results. Includes calculating bulk density,
    voidage, Hausner ratio, and compression index. Optionally plots the results.

    **Parameters:**
    - `plot` (bool): If True, generates plots of bulk density and voidage over time.
    - `bulk_dens_method` (str): Method to calculate bulk density. Options are 'precise' or 'sample'.
            Precises uses a more accurate method based on particle positions, considering cuttoffs
            while 'sample' uses a sampling method based on a fraction of the domain.
    """
    global study, project

    time_set = study.GetTimeSet()
    timeset_arr = time_set.GetValues()
    settled_timestep = 0
    particles = study.GetParticles()

    if bulk_dens_method == 'precise':
        uncompr_dens, voidage = _calc_bulk_dens_v2(particles, 0)
        compr_dens, voidage = _calc_bulk_dens_v2(particles, -1)
    elif bulk_dens_method == 'coarse':
        bulk_dens = _calc_bulk_dens(particles, settled_timestep)
        compr_dens = _calc_bulk_dens(particles, -1)

    hausner_ratio = compr_dens / uncompr_dens
    compr_idx = 100 * (compr_dens - uncompr_dens) / compr_dens

    bulk_dens = []
    voidage = []
    if plot:
        PLOTS_DIR = os.path.join(PROJECT_DIR, 'plots')
        for timestep in np.nditer(time_set, flags=['refs_ok']):
            bulk_dens_ts, voidage_ts = _calc_bulk_dens_v2(
                particles, timestep.item()
            )
            bulk_dens.append(bulk_dens_ts)
            voidage.append(voidage_ts)
        bulk_dens_t = np.array(bulk_dens)
        voidage_t = np.array(voidage)

        # Create a plot with two y-axes
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot bulk density on the left y-axis
        color = 'tab:blue'
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Bulk Density (kg/m³)', color=color)
        ax1.plot(timeset_arr, bulk_dens_t, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        # Create a second y-axis for voidage
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Voidage', color=color)
        ax2.plot(timeset_arr, voidage_t, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        # Add title and grid
        plt.title('Bulk Density and Voidage vs Time')
        ax1.grid(True, alpha=0.3)

        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

        fig.tight_layout()
        plt.savefig(
            os.path.join(
                PLOTS_DIR,
                'bulk_density_voidage.png'),
            dpi=300)

    col_vals = [
        P_RADIUS, P_DENSITY, P_YOUNGMOD, P_POISSON,
        PP_DYNAMIC_FRICTION, PP_STATIC_FRICTION, PP_COR,
        PW_DYNAMIC_FRICTION, PW_STATIC_FRICTION, PW_COR,
        ROLLING_FRICTION, COMPR_PRESSURE,
        NORMAL_FORCE_MODEL, TANGENTIAL_FORCE_MODEL, ADHESION_MODEL,
        ROLLING_MODEL, PARTICLE_BOX_LEN, uncompr_dens, compr_dens,
        hausner_ratio, compr_idx
    ]

    col_names = [
        'p_radius', 'p_density', 'p_youngmod', 'p_poisson',
        'pp_dynamic_friction', 'pp_static_friction', 'pp_cor',
        'pw_dynamic_friction', 'pw_static_friction', 'pw_cor',
        'rolling_friction', 'compression_pressure',
        'normal_force_model', 'tangential_force_model', 'adhesion_model',
        'rolling_model', 'box_len', 'bulk_density', 'compressed_density',
        'hausner_ratio', 'compression_index'
    ]

    global particle_warning, particle_rng
    particle_warning = False
    particle_rng = []
    
    if particles.GetNumberOfParticles(time_set[0]) != particles.GetNumberOfParticles(
        time_set[-1]):
            warnings.warn(
                "Particles are being lost during the simulation."
                "Results set to NaN."
            )
            particle_warning = True
            particle_rng = [
                particles.GetNumberOfParticles(time_set[0]), 
                particles.GetNumberOfParticles(time_set[-1])
            ]

    assert len(col_vals) == len(col_names), \
        "Column values and names must have the same length."

    # Write results to a CSV file
    with open('results.csv', 'w') as f:
        # Convert col_names to a comma-separated string
        f.write(','.join(col_names))
        f.write('\n')
        # Convert cols to a comma-separated string
        f.write(','.join(map(str, col_vals)))

    # Dump the results to a SQLite database
    sqlite_path = os.path.abspath('../results.db')
    with sqlite3.connect(sqlite_path) as conn:
        cursor = conn.cursor()

        create_table_query = '''CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            p_radius REAL,
            p_density REAL,
            p_youngmod REAL,
            p_poisson REAL,
            pp_dynamic_friction REAL,
            pp_static_friction REAL,
            pp_cor REAL,
            pw_dynamic_friction REAL,
            pw_static_friction REAL,
            pw_cor REAL,
            rolling_friction REAL,
            compression_pressure REAL,
            normal_force_model TEXT,
            tangential_force_model TEXT,
            adhesion_model TEXT,
            rolling_model TEXT,
            box_len REAL,
            bulk_density REAL,
            compressed_density REAL,
            hausner_ratio REAL,
            compression_index REAL
        )'''
        insert_query = f'''INSERT INTO results (
            p_radius, p_density, p_youngmod, p_poisson,
            pp_dynamic_friction, pp_static_friction, pp_cor,
            pw_dynamic_friction, pw_static_friction, pw_cor,
            rolling_friction, compression_pressure,
            normal_force_model, tangential_force_model, adhesion_model,
            rolling_model, box_len, bulk_density, compressed_density,
            hausner_ratio, compression_index
        ) VALUES ({','.join(['?']*len(col_vals))})
        '''

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
    "particle_path": "{{PARTICLE_PATH}}"
}

setup()
load_meshes()
load_material_properties()
load_interactions()
gen_particle(shape_dict)
sim_physics()
insertion_settings()
set_domain_settings()
settle_particles()
compress_particles()
post_process()

if particle_warning:
    raise RuntimeWarning(
        f"Particles were lost during the simulation. "
        f"Initial particle count: {particle_rng[0]}, "
        f"Final particle count: {particle_rng[1]}. "
        f"Check the simulation settings and results."
    )
 