#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt


# ========== WORKSPACE VARS ==========
project_dir: str = os.getcwd()

bb_rocky_path = '/rds/bear-apps/2023a/EL8-ice/software/ANSYS_Rocky/2024R2.0/bin/Rocky'
vm_rocky_path = '/home/rocky-vm/ansys_inc/v242/rocky/bin/Rocky'


# ========== RUN BASED VARS ==========
"""
This is where you can set the parameters for the simulation.
"""
# ---------- Particle Vars ----------
p_radius: float = {{RADIUS_P}}  # m
p_density: float = {{DENSITY_P}}  # kg/m^3
p_youngmod: float = {{YOUNGMOD_P}}  # Pa
p_poisson: float = {{POISSON_P}}  # Poisson ratio

# ---------- Interaction Vars -------------

rolling_model = '{{ROLLING_MODEL}}'  # 'type_1', 'type_3', 'none', 'custom'
assert rolling_model in ['type_1', 'type_3', 'none', 'custom']

pp_dynamic_friction: float = {{DYNAMIC_FRICTION_PP}}
pp_static_friction: float = {{STATIC_FRICTION_PP}}
pp_cor: float = {{COR_PP}}
pp_rolling_friction: float = {{ROLLING_FRICTION_PP}}

pw_dynamic_friction: float = {{DYNAMIC_FRICTION_PW}}
pw_static_friction: float = {{STATIC_FRICTION_PW}}
pw_cor: float = {{COR_PW}}
pw_rolling_friction: float = {{ROLLING_FRICTION_PW}}

# ---------- Contact model --------
normal_force_model = '{{NORMAL_MODEL}}'
assert normal_force_model in ['linear_hysteresis', 'linear_elastic_viscous', 'damped_hertzian', 'custom']

tangential_force_model = '{{TANG_MODEL}}'
assert tangential_force_model in ['elastic_coulomb', 'coulomb_limit', 'mindlin_deresiewicz', 'custom']

adhesion_model = '{{ADH_MODEL}}'
assert adhesion_model in ['none', 'constant', 'linear', 'JKR', 'custom']

# ---------- Sim Vars -------------
particle_box_len: float = {{L_BOX}}  # m
t_fill: float = 3  # s
t_settle: float = 1  # s

compr_pressure: float = {{P_COMPRESS}}  # Pa
t_compression: float = 1  # s

# insert type
insert_type: str = 'vol'  # 'vol', 'ins'
assert insert_type in ['vol', 'ins']

#  ========= SOLVER VARS ==============
nprocs: int = 20
neighbour_search: str = None
if neighbour_search is not None:
    assert neighbour_search in ['BVH', 'RegularGrid', 'SparseGrid']


processor: str = 'CPU'
assert processor in ['CPU', 'GPU', 'MULTI_GPU']

runtime: float = 5.  # s

# ==========Rocky Setup==========
# ===============================

# UNCOMMENT IF USING PYROCKY

# rocky = pyrocky.launch_rocky(
#     headless=True,
#     rocky_exe=bb_rocky_path,
# )

# rocky = pyrocky.connect_to_rocky()

# project = rocky.api.CreateProject()

project = app.CreateProject()
project.SaveProject(
    os.path.join(project_dir, 'uniaxial_compression.rocky')
)
study = project.GetStudy()
study.SetName('Uniaxial Compression')

# ========== Mesh Generation ==========
# =====================================

meshdir = '{{MESH_DIR}}'
compr_wall1_stl_path = os.path.join(meshdir, 'compressive_wall1.stl')
compr_wall2_stl_path = os.path.join(meshdir, 'compressive_wall2.stl')
inlet_stl_path = os.path.join(meshdir, 'insert.stl')
particle_box_stl_path = os.path.join(meshdir, 'particlebox.stl')

# Put compressing wall 0.5 times the size of the particle box
compr_wall1 = study.ImportWall(compr_wall1_stl_path,
                               import_scale=1.0,
                               convert_yz=True)[0]
compr_wall1.SetName('Compression Wall 1')

# Put other compressing at edge of the particle box
compr_wall2 = study.ImportWall(compr_wall2_stl_path,
                               import_scale=1.0,
                               convert_yz=True)[0]
compr_wall2.SetName('Compression Wall 2')
compr_wall2.SetTranslation([particle_box_len/2 + 1e-6, 0, 0])

# Insert particle box
particle_box = study.ImportWall(
    particle_box_stl_path,
    import_scale=1.0,
    convert_yz=True)[0]
particle_box.SetName('Particle Box')
if insert_type == 'ins':
    particle_box.SetDisableTime(t_fill + t_settle)
    # Set the particle box to be shifted 1 micron up
    # to prevent ineraction with periodic boundaries
    particle_box.SetTranslation([0, 1e-6, 0])
else:
    particle_box.SetDisableTime(t_settle)

# Insert inlet plane
insert_inlet = study.ImportSurface(
    inlet_stl_path,
    import_scale=0.99,
    convert_yz=True)[0]
insert_inlet.SetName('Insert Inlet')


# ========== Materials Definition ==========
# ==========================================

material_collection = study.GetMaterialCollection()
material_collection.Clear()

particle_mat = material_collection.AddSolidMaterial()
particle_mat.SetName("Particle Material")
particle_mat.SetDensity(p_density, 'kg/m3')
particle_mat.SetYoungsModulus(p_youngmod, 'Pa')
particle_mat.SetPoissonRatio(p_poisson)
particle_mat.SetUseBulkDensity(False)

wall_mat = material_collection.AddSolidMaterial()
wall_mat.SetName("Wall Material")
wall_mat.SetDensity(2700, 'kg/m3')
wall_mat.SetYoungsModulus(5e6, 'Pa')
wall_mat.SetPoissonRatio(0.3)
wall_mat.SetUseBulkDensity(False)

# Set the material for the meshes
particle_box.SetMaterial(wall_mat)
compr_wall1.SetMaterial(wall_mat)
compr_wall2.SetMaterial(wall_mat)

# =========== Interaction Properties ==========
# ============================================

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
pp_interaction.SetRestitutionCoefficient(pp_cor)
pp_interaction.SetStaticFriction(pp_static_friction)
pp_interaction.SetDynamicFriction(pp_dynamic_friction)
if rolling_model != 'none':
    pp_interaction.SetRollingFriction(pp_rolling_friction)

# Set the contact laws for the particle-wall interaction
pw_interaction.SetRestitutionCoefficient(pw_cor)
pw_interaction.SetStaticFriction(pw_static_friction)
pw_interaction.SetDynamicFriction(pw_dynamic_friction)
if rolling_model != 'none':
    pw_interaction.SetRollingFriction(pw_rolling_friction)


# ========== Particle Sizes ==========
# ====================================
particle = study.CreateParticle()
size_distr_lst = particle.GetSizeDistributionList()
size_distr_lst.Clear()  # clear auto-generated size distribution

psd = size_distr_lst.New()
psd.SetSize(p_radius, 'm')
psd.SetCumulativePercentage(100)
particle.SetMaterial(particle_mat)

if isinstance(p_radius, float) or isinstance(p_radius, int):
    particle = study.CreateParticle()
    size_distr_lst = particle.GetSizeDistributionList()
    size_distr_lst.Clear()

    psd = size_distr_lst.New()
    psd.SetSize(p_radius, 'm')
    psd.SetCumulativePercentage(100)

# If it is a dictionary, create a particle size distribution
# with multiple sizes
elif isinstance(p_radius, dict):
    # Check if the values are valid
    if sum(p_radius.values()) == 1:
        p_radius = {k: v * 100 for k, v in p_radius.items()}
    elif sum(p_radius.values()) == 100:
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
    sorted_dict = dict(sorted(p_radius.items(), reverse=True))
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

# ========== Simulation Physics ==========
# ========================================
# Set the physics for the simulation
physics = study.GetPhysics()
physics.SetNormalForceModel(normal_force_model)
physics.SetTangentialForceModel(tangential_force_model)
physics.SetAdhesionModel(adhesion_model)

# ========== Insertion Settings ==========
# ========================================

fill_box_vol = particle_box_len**3  # m^3
particle_vol = (4/3) * np.pi * p_radius**3  # m^3

# 0.64 is an avg packing fraction of sphereical particles
if insert_type == 'ins':
    n_particles = np.rint(
        fill_box_vol * 0.64 / particle_vol
    ).astype(int).item()

    mass_particles = particle_vol * p_density * n_particles
    flowr = mass_particles / t_fill

    particle_inlet = study.CreateParticleInlet(
        insert_inlet, particle
    )

    input_property_lst = particle_inlet.GetInputPropertiesList()
    input_property_lst[0].SetMassFlowRate(flowr, 'kg/s')

    particle_inlet.SetStartTime(0)
    particle_inlet.SetStopTime(t_fill)
    particle_inlet.DisablePeriodic()
else:
    fill_box_vol = particle_box_len**3  # m^3
    particle_vol = (4/3) * np.pi * p_radius**3  # m^3
    n_particles = fill_box_vol / particle_vol
    mass_particles = particle_vol * p_density * n_particles


# ========== Compression Motion ==========
# ========================================

frame_source = study.GetMotionFrameSource()
compr_motion_frame = frame_source.NewFrame()
motions = compr_motion_frame.GetMotions()

# Handling the free body motion
freebody_motion = motions.New()
freebody_motion.SetType('Free Body Translation')
freebody = freebody_motion.GetTypeObject()
freebody.SetFreeMotionDirection('x')
if insert_type == 'ins':
    freebody.SetStartTime(t_fill+t_settle)
else:
    freebody_motion.SetStartTime(t_settle)

# Set the compression wall motion
force_magnitude = compr_pressure * particle_box_len**2
force_motion = motions.New()
force_motion.SetType('Additional Force')
add_force = force_motion.GetTypeObject()
add_force.SetForceValue([-force_magnitude, 0, 0], 'N')
if insert_type == 'ins':
    force_motion.SetStartTime(t_fill+t_settle)
else:
    force_motion.SetStartTime(t_settle)

compr_motion_frame.ApplyTo(compr_wall1)

# ========== Simulation Settings ==========
# =========================================

study = project.GetStudy()
solver = study.GetSolver()
solver.SetNumberOfProcessors(int(nprocs))


solver.SetSimulationDuration(runtime, 's')
solver.SetReleaseParticlesWithoutOverlapCheck(True)

project.SaveProject()
# rocky.close()

print(f"Running simulation with {processor} solver.")
study.StartSimulation()

while study.IsSimulating():
    study.RefreshResults()
    print(f"Simulation Progress: {study.GetProgress():.2f} %")
print("Simulation completed.")

# ========== Post Processing ==========
# =====================================

plot: bool = True

if plot:
    def plot_mass(t, mass, plot_name):
        plt.plot(t, mass)
        plt.ylabel("Mass (kg)")
        plt.xlabel("Time (s)")
        plt.savefig(f'plots/{plot_name}.png')

print("Post-processing simulation results...")
# Add code to post-process the simulation results here
# Get particle data
particles = study.GetParticles()
x = particles.GetGridFunction('Coordinate : X')

# Find settled time step
timestep = study.GetTimeSet()
settled_timestep = np.where(timestep == 2)[0][0].item()

x_arr_init = x.GetArray(time_step=settled_timestep)
x_max_init, x_min_init = x_arr_init.min().item(), x_arr_init.max().item()

x_arr_compr = x.GetArray(time_step=-1)
x_max_compr = x_arr_compr.min().item()
x_min_compr = x_arr_compr.max().item()

y = particles.GetGridFunction('Coordinate : Y')
y_arr_init = y.GetArray(time_step=settled_timestep)
y_max_init, y_min_init = y_arr_init.min().item(), y_arr_init.max().item()

y_arr_compr = y.GetArray(time_step=-1)
y_max_compr = y_arr_compr.min().item()
y_min_compr = y_arr_compr.max().item()
z = particles.GetGridFunction('Coordinate : Z')
z_arr_init = z.GetArray(time_step=settled_timestep)
z_max_init = z_arr_init.min().item()
z_min_init = z_arr_init.max().item()

z_arr_compr = z.GetArray(time_step=-1)
z_max_compr = z_arr_compr.min().item()
z_min_compr = z_arr_compr.max().item()
processes = project.GetUserProcessCollection()

cuboid_selection_init = processes.CreateCubeProcess(particles)

cuboid_selection_init.SetSize(
    x_max_init-x_min_init,
    y_max_init-y_min_init,
    z_max_init-z_min_init,
    unit="m"
)

cuboid_selection_init.SetCenter(
    (x_max_init+x_min_init)/2,
    (y_max_init+y_min_init)/2,
    (z_max_init+z_min_init)/2,
    unit="m"
)

t, mass_init = cuboid_selection_init.GetNumpyCurve('Particles Mass')
plot_mass(t, mass_init, "mass_init") if plot else None

cuboid_selection_compr = processes.CreateCubeProcess(particles)

cuboid_selection_compr.SetSize(
    x_max_compr-x_min_compr,
    y_max_compr-y_min_compr,
    z_max_compr-z_min_compr,
    unit="m"
)

cuboid_selection_init.SetCenter(
    (x_max_compr+x_min_compr)/2,
    (y_max_compr+y_min_compr)/2,
    (z_max_compr+z_min_compr)/2,
    unit="m"
)

t, mass_compr = cuboid_selection_compr.GetNumpyCurve('Particles Mass')
plot_mass(t, mass_compr, "mass_compressed") if plot else None

V_bulk = (x_max_init-x_min_init
          )*(y_max_init-y_min_init
             )*(z_max_init-z_min_init)
bulk_dens = np.abs(mass_init.max()/V_bulk).item()

V_packed = (x_max_compr-x_min_compr
            )*(y_max_compr-y_min_compr
               )*(z_max_compr-z_min_compr)
packed_dens = np.abs(mass_compr.max()/V_packed).item()
print(f"Packed density: {packed_dens} kg/m^3")

hausner_ratio = bulk_dens/packed_dens
print("Hausner Ratio: ", hausner_ratio)
compr_indx = 100 * (1-packed_dens/bulk_dens)
print("Compressibility Index: ", compr_indx)

cols = [
    p_radius,
    p_density,
    p_poisson,
    p_youngmod,
    pp_dynamic_friction,
    pp_static_friction,
    pp_rolling_friction,
    pp_cor,
    pw_dynamic_friction,
    pw_static_friction,
    pw_rolling_friction,
    pw_cor,
    particle_box_len,
    compr_pressure,
    normal_force_model,
    tangential_force_model,
    rolling_model,
    adhesion_model
]

col_names = [
    'p_radius',
    'p_density',
    'p_poisson',
    'p_youngmod',
    'pp_dynamic_friction',
    'pp_static_friction',
    'pp_rolling_friction',
    'pp_cor',
    'pw_dynamic_friction',
    'pw_static_friction',
    'pw_rolling_friction',
    'pw_cor',
    'particle_box_len',
    'compr_pressure',
    'normal_force_model',
    'tangential_force_model',
    'rolling_model',
    'adhesion_model',
    'particle_vol'
]
with open('results.txt', 'w') as f:
    f.write(col_names)
    f.write('\n')
    f.write(str(cols))