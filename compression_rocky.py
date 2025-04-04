#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Abhirup Roy

import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
# import gmsh
import ansys.rocky.core as pyrocky

from compr_meshgen import create_compr_walls, create_insert, create_particlebox

"""
TODO:
- Fix insertion mass!
- Check errors on BB
    - Do NOT use Periodic @ geometry lims
- Validate results

- Set up param sweep
"""


class UniaxialCompression:
    """
    Class to perform uniaxial compression tests using Rocky DEM.
    """
    def __init__(self, **kwargs):
        """
        Initialize the UniaxialCompression class

        Parameters
        ----------
        - project_dir : str
            The working directory for the simulation.
            Default is the current working directory.
        - connection_type : str
            The type of connection to Rocky. Options are 'launch' or 'connect'.
            Default is 'launch'.
        - rocky_exe : str
            The path to the Rocky executable on Linux VM - use 'VM' for this
            '/home/rocky-vm/ansys_inc/v242/rocky/bin/Rocky'
            For BlueBear default use 'BB' - this corresponds to
            '/rds/bear-apps/2023a/EL8-ice/software/ANSYS_Rocky/2024R2.0/bin/Rocky'
        - radius : float
            The radius of the cylinder in meters. Default is 150e-6 m.
        - mesh_stl_names : list
            A list of mesh STL filenames. Default is None.
        - density : float
            The density of the particles in kg/m^3. Default is 2700 kg/m^3.
        - particle_box_len : float
            The length of the particle box in meters. Default is 0.01 m.

        """
        self.connection_type = kwargs.get('connection_type', 'launch')
        self.rocky_exe = kwargs.get('rocky_exe', None)
        if self.rocky_exe == 'BB':
            self.rocky_exe = '/rds/bear-apps/2023a/EL8-ice/software/ANSYS_Rocky/2024R2.0/bin/Rocky'
        elif self.rocky_exe == 'VM':
            self.rocky_exe = '/home/rocky-vm/ansys_inc/v242/rocky/bin/Rocky'
        else:
            if not self.rocky_exe:
                pass
            elif not os.path.exists(self.rocky_exe):
                raise FileNotFoundError(
                    f"Rocky executable not found at {self.rocky_exe}.",
                    "Please provide a valid path."
                )
            else:
                self.rocky_exe = os.path.abspath(self.rocky_exe)
        self.headless = bool(kwargs.get('headless', True))

        # Get the directory of the simulation
        self.project_dir: str = kwargs.get('working_dir', os.getcwd())
        self.project_dir = os.path.abspath(self.project_dir)

        # Loading particle parameters
        self.p_radius: float = kwargs.get('p_radius', 150e-6)  # m
        self.p_density: float = kwargs.get('p_density', 2700)  # kg/m^3
        self.p_youngmod: float = kwargs.get('p_young_mod', 5e6)  # Pa
        self.p_poisson: float = kwargs.get('p_poisson', 0.3)

        # Filling parameters
        self.particle_box_len: float = kwargs.get(
            'particle_box_len', 10/1000)  # m
        self.t_fill = kwargs.get('t_fill', 3)  # s
        self.t_settle = kwargs.get('t_settle', 1)  # s
        # Attributes for easy access
        self.meshes = {}
        self.materials = {}

        self.compr_pressure = kwargs.get('pressure', 15e3)  # Pa
        self.t_compression = kwargs.get('t_compression', 1)  # s
        # Call the setup functions
        print("Setting up the simulation...")
        self._meshgen()
        self._setup()
        self._load_meshes()
        self._set_materials()
        self._set_particle_size()
        self._domain_settings()
        self._insertion_settings()
        self._compress_wall()

    def _setup(self):
        """
        Setup the simulation case
        """
        if self.connection_type == 'launch':
            self.rocky = pyrocky.launch_rocky(
                headless=self.headless,
                rocky_exe=self.rocky_exe
            )
        elif self.connection_type == 'connect':
            # Assume Rocky is already running in Pyrocky mode
            # (i.e run `Rocky --pyrocky`)
            self.rocky = pyrocky.connect()

        self.project = self.rocky.api.CreateProject()
        self.project.SaveProject(
            os.path.join(self.project_dir, 'uniaxial_compression.rocky')
        )
        self.study = self.project.GetStudy()
        self.study.SetName('Uniaxial Compression')

    def _meshgen(self):
        """
        Generate the meshes for the simulation
        """
        # Create the meshes
        create_particlebox(self.particle_box_len)
        create_compr_walls(self.particle_box_len)
        create_insert(self.particle_box_len)

    def _load_meshes(self):
        """
        Define the parameters for the simulation
        """
        meshdir = os.path.abspath('meshes')
        compr_wall1_stl_path = os.path.join(meshdir, 'compressive_wall1.stl')
        compr_wall2_stl_path = os.path.join(meshdir, 'compressive_wall2.stl')
        inlet_stl_path = os.path.join(meshdir, 'insert.stl')
        particle_box_stl_path = os.path.join(meshdir, 'particlebox.stl')

        # Put compressing wall 0.5 times the size of the particle box
        compr_wall1 = self.study.ImportWall(compr_wall1_stl_path,
                                            import_scale=1.0,
                                            convert_yz=True)[0]
        compr_wall1.SetName('Compression Wall 1')

        # Put other compressing at edge of the particle box
        compr_wall2 = self.study.ImportWall(compr_wall2_stl_path,
                                            import_scale=1.0,
                                            convert_yz=True)[0]
        compr_wall2.SetName('Compression Wall 2')
        compr_wall2.SetTranslation([self.particle_box_len/2 + 1e-6, 0, 0])

        # Insert particle box
        particle_box = self.study.ImportWall(
            particle_box_stl_path,
            import_scale=1.0,
            convert_yz=True)[0]
        particle_box.SetName('Particle Box')
        particle_box.SetDisableTime(self.t_fill+self.t_settle)
        # Set the particle box to be shifted 1 micron up
        # to prevent ineraction with periodic boundaries
        particle_box.SetTranslation([0, 1e-6, 0])

        # Insert inlet plane
        insert_inlet = self.study.ImportSurface(
            inlet_stl_path,
            import_scale=0.99,
            convert_yz=True)[0]
        insert_inlet.SetName('Insert Inlet')

        self.meshes["box"] = particle_box
        self.meshes["compr_wall1"] = compr_wall1
        self.meshes["compr_wall2"] = compr_wall2
        self.meshes["insert_inlet"] = insert_inlet

    def _set_materials(self):
        """
        Define the particle parameters
        """
        material_collection = self.study.GetMaterialCollection()
        material_collection.Clear()

        particle_mat = material_collection.AddSolidMaterial()
        particle_mat.SetName("Particle Material")
        particle_mat.SetDensity(self.p_density, 'kg/m3')
        particle_mat.SetYoungsModulus(self.p_youngmod, 'Pa')
        particle_mat.SetPoissonRatio(self.p_poisson)
        particle_mat.SetUseBulkDensity(False)

        wall_mat = material_collection.AddSolidMaterial()
        wall_mat.SetName("Wall Material")
        wall_mat.SetDensity(2700, 'kg/m3')
        wall_mat.SetYoungsModulus(5e6, 'Pa')
        wall_mat.SetPoissonRatio(0.3)
        wall_mat.SetUseBulkDensity(False)

        # Set the material for the meshes
        print(self.meshes.keys())
        for mesh in list(self.meshes.values())[:-1]:
            mesh.SetMaterial(wall_mat)

        # Save the material collection
        self.materials["particle_mat"] = particle_mat
        self.materials["wall_mat"] = wall_mat

    def _set_particle_size(self):
        """
        Set the particle size and density
        """
        self.particle = self.study.CreateParticle()
        size_distr_lst = self.particle.GetSizeDistributionList()
        size_distr_lst.Clear()  # clear auto-generated size distribution

        psd = size_distr_lst.New()
        psd.SetSize(self.p_radius, 'm')
        psd.SetCumulativePercentage(100)

        # Set the particle material
        self.particle.SetMaterial(self.materials["particle_mat"])

    def _domain_settings(self):
        self.domain = self.study.GetDomainSettings()
        self.domain.SetDomainType('CARTESIAN')
        self.domain.EnableUseBoundaryLimits()
        self.domain.DisablePeriodicAtGeometryLimits()
        # Set periodic limits in X and Y directions
        # no periodic in Z-direction to allow compression
        self.domain.SetCartesianPeriodicDirections('XY')
        self.domain.SetPeriodicLimitsMaxCoordinates(
            [self.particle_box_len/2, self.particle_box_len/2, np.inf])
        self.domain.SetPeriodicLimitsMinCoordinates(
            [-self.particle_box_len/2, -self.particle_box_len/2, -np.inf])

    def _insertion_settings(self):
        fill_box_vol = self.particle_box_len**3 #m^3
        particle_vol = (4/3) * np.pi * self.p_radius**3 # m^3

        # 0.64 is an avg packing fraction of sphereical particles

        n_particles = np.rint(
            fill_box_vol * 0.64 / particle_vol
        ).astype(int).item()
        
        mass_particles = particle_vol * self.p_density * n_particles
        flowr = mass_particles / self.t_fill

        insert_inlet = self.meshes["insert_inlet"]
        particle_inlet = self.study.CreateParticleInlet(
            insert_inlet, self.particle
        )

        input_property_lst = particle_inlet.GetInputPropertiesList()
        input_property_lst[0].SetMassFlowRate(flowr, 'kg/s')

        particle_inlet.SetStartTime(0)
        particle_inlet.SetStopTime(self.t_fill)
        particle_inlet.DisablePeriodic()

    def _compress_wall(self):
        frame_source = self.study.GetMotionFrameSource()
        compr_motion_frame = frame_source.NewFrame()
        motions = compr_motion_frame.GetMotions()

        # Handling the free body motion
        freebody_motion = motions.New()
        freebody_motion.SetType('Free Body Translation')
        freebody = freebody_motion.GetTypeObject()
        freebody.SetFreeMotionDirection('x')
        freebody_motion.SetStartTime(2)

        # Set the compression wall motion
        force_magnitude = self.compr_pressure * self.particle_box_len**2
        force_motion = motions.New()
        force_motion.SetType('Additional Force')
        add_force = force_motion.GetTypeObject()
        add_force.SetForceValue([-force_magnitude, 0, 0], 'N')
        force_motion.SetStartTime(2)

        compressing_wall = self.meshes["compr_wall1"]
        compr_motion_frame.ApplyTo(compressing_wall)

    def simulate(self,
                 processor: str = "cpu",
                 nproc: int = 12,
                 runtime: float = 5,
                 **kwargs):

        """
        Run the simulation with the specified parameters.

        Parameters
        ----------
        - proc : str
            The type of solver to use (not case-sensitive). 
            Options are 'cpu', 'gpu', or 'multi_gpu'.
            Default is 'cpu'.
        - nproc : int
            The number of processors to use. Default is 12.
        - runtime : float
            The total simulation time in seconds. Default is 5 seconds

        Keyword Arguments
        -----------------
        - neighbour_search : str
            The neighbour search model to use.
            Options are 'BVH', 'RegularGrid', or 'SparseGrid'.
            Default is None.
        - target_gpu : int|str
            The GPU to use if the solver type is 'gpu'. Default is None.
        - target_gpus : list
            The GPUs to use if the solver type is 'multi_gpu'. Default is None.
        """
        self.solver = self.study.GetSolver()
        self.solver.SetNumberOfProcessors(int(nproc))

        neighbour_search: str = kwargs.get('neighbour_search', None)
        # delete_results: bool = kwargs.get('delete_results', False)

        # Handle the solver type - allow lowercase inputs as well
        _proc = processor.upper()

        if _proc in ['CPU', 'GPU', 'MULTI_GPU']:
            self.solver.SetSimulationTarget(_proc)

            if _proc == 'GPU':
                target_gpu = kwargs.get('target_gpu')
            elif _proc == 'MULTI_GPU':
                target_gpus: list = kwargs.get('target_gpus')

                warnings.warn(
                    "GPU solver is not fully tested yet. Use with caution."
                )

        else:
            raise ValueError(
                f"Unknown solver type: {_proc}. Use 'CPU', 'GPU', or 'MULTI_GPU'.")

        self.solver.SetSimulationDuration(runtime, 's')
        self.solver.SetReleaseParticlesWithoutOverlapCheck(True)
        
        if neighbour_search:
            if neighbour_search in ['BVH', 'RegularGrid', 'SparseGrid']:
                self.solver.SetNeighborSearchModel(neighbour_search)
            else:
                raise ValueError(f"Unknown neighbour search model: {neighbour_search}. Use 'BVH', 'RegularGrid', or 'SparseGrid'.")

        print(f"Running simulation with {processor} solver.")
        self.study.StartSimulation()
        print("Simulation starting...")
        while self.study.IsSimulating():
            self.study.RefreshResults()
            print(f"Simulation Progress: {self.study.GetProgress():.2f} %")
        print("Simulation completed.")

    def postprocess(self, **kwargs):
        """
        Post-process the simulation results
        """
        plot: bool = kwargs.get('plot', True)
        if plot:
            def plot_mass(t, mass, plot_name):
                plt.plot(t, mass)
                plt.ylabel("Mass (kg)")
                plt.xlabel("Time (s)")
                plt.savefig(f'plots/{plot_name}.png')

        print("Post-processing simulation results...")
        # Add code to post-process the simulation results here
        # Get particle data
        self.particles = self.study.GetParticles()
        x = self.particles.GetGridFunction('Coordinate : X')

        # Find settled time step
        timestep = self.study.GetTimeSet()
        settled_timestep = np.where(timestep == 2)[0][0].item()

        x_arr_init = x.GetArray(time_step=settled_timestep)
        x_max_init, x_min_init = x_arr_init.min().item(), x_arr_init.max().item()

        x_arr_compr = x.GetArray(time_step=-1)
        x_max_compr = x_arr_compr.min().item()
        x_min_compr = x_arr_compr.max().item()

        y = self.particles.GetGridFunction('Coordinate : Y')
        y_arr_init = y.GetArray(time_step=settled_timestep)
        y_max_init, y_min_init = y_arr_init.min().item(), y_arr_init.max().item()

        y_arr_compr = y.GetArray(time_step=-1)
        y_max_compr = y_arr_compr.min().item()
        y_min_compr = y_arr_compr.max().item()
        z = self.particles.GetGridFunction('Coordinate : Z')
        z_arr_init = z.GetArray(time_step=settled_timestep)
        z_max_init = z_arr_init.min().item()
        z_min_init = z_arr_init.max().item()

        z_arr_compr = z.GetArray(time_step=-1)
        z_max_compr = z_arr_compr.min().item()
        z_min_compr = z_arr_compr.max().item()
        self.processes = self.project.GetUserProcessCollection()

        cuboid_selection_init = self.processes.CreateCubeProcess(self.particles)

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

        cuboid_selection_compr = self.processes.CreateCubeProcess(self.particles)

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
        print(f"Bulk density: {bulk_dens} kg/m^3")

        V_packed = (x_max_compr-x_min_compr
                    )*(y_max_compr-y_min_compr
                    )*(z_max_compr-z_min_compr)
        packed_dens = np.abs(mass_compr.max()/V_packed).item()
        print(f"Packed density: {packed_dens} kg/m^3")

        hausner_ratio = bulk_dens/packed_dens
        print("Hausner Ratio: ", hausner_ratio)
        compr_indx = 100 * (1-packed_dens/bulk_dens)
        print("Compressibility Index: ", compr_indx)


if __name__ == "__main__":
    uniax = UniaxialCompression(
        connection_type='launch',
        rocky_exe='BB',
        headless=True
    )
    uniax.simulate(
        processor='cpu',        
        nproc=10, # cores
        runtime=5  # seconds
    )
    uniax.postprocess(
        plot=True
    )
