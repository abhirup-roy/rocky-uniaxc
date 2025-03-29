#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Abhirup Roy

import os
import numpy as np
import ansys.rocky.core as pyrocky

"""
TODO:
- Select correct compr wall coordinates
- verify the compression direction is right
- assign the correct material to the walls and particles
- ensure the simulation sets up correctly
- add functionality to run the simulation

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
        - radius : float
            The radius of the cylinder in meters. Default is 150e-6 m.
        - mesh_stl_names : list
            A list of mesh STL filenames. Default is None.
        - density : float
            The density of the particles in kg/m^3. Default is 2700 kg/m^3.
        - particle_box_len : float
            The length of the particle box in meters. Default is 0.01 m.

        """
        # Get the directory of thhe simulation
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
        self.t_fill = kwargs.get('t_fill', 1)  # s

        self.compr_pressure = kwargs.get('pressure', 15e3)  # Pa
        self.t_compression = kwargs.get('t_compression', 1)  # s
        # Call the setup functions
        self._setup()
        self._load_meshes()
        self._set_materials()

    def _setup(self):
        """
        Setup the simulation case
        """

        self.rocky = pyrocky.launch_rocky(headless=True)
        self.project = self.rocky.api.CreateProject()
        self.project.SaveProject(
            os.path.join(self.project_dir, 'uniaxial_compression.rocky')
        )
        self.study = self.project.CreateStudy()
        self.study.SetName('Uniaxial Compression')

    def _load_meshes(self):
        """
        Define the parameters for the simulation
        """
        # Define abs path for the meshes
        meshdir = os.path.join(self.project_dir, 'meshes')
        for f in os.listdir(meshdir):
            if f == 'insert_base.stl':
                base_stl_path = os.path.join(meshdir, f)
            elif f == 'insert_support1.stl':
                support1_stl_path = os.path.join(meshdir, f)
            elif f == 'insert_support2.stl':
                support2_stl_path = os.path.join(meshdir, f)
            elif f == 'insert_inlet.stl':
                inlet_stl_path = os.path.join(meshdir, f)
            elif f == 'square_wall_negZ.stl':
                support3_stl_path = os.path.join(meshdir, f)
                compr_wall1_stl_path = os.path.join(meshdir, f)
            elif f == 'square_wall_posZ.stl':
                support4_stl_path = os.path.join(meshdir, f)
                compr_wall2_stl_path = os.path.join(meshdir, f)

        # Insert compressing walls
        compr_wall1 = self.study.ImportWall(compr_wall1_stl_path,
                                            import_scale=1.0,
                                            convert_yz=True)[0]
        compr_wall1.SetName('Compression Wall 1')
        compr_wall1.SetDisableTime(2, 's')
        compr_wall2 = self.study.ImportWall(compr_wall2_stl_path,
                                            import_scale=1.0,
                                            convert_yz=True)[0]
        compr_wall2.SetName('Compression Wall 2')
        compr_wall2.SetDisableTime(2, 's')

        # Insert insertion support meshes
        insert_base = self.study.ImportWall(base_stl_path,
                                            import_scale=self.particle_box_len,
                                            convert_yz=True)[0]

        insert_base.SetName('Insert Base')
        insert_base.SetDisableTime(2, 's')

        insert_support1 = self.study.ImportWall(
            support1_stl_path,
            import_scale=self.particle_box_len,
            convert_yz=False)[0]
        insert_support1.SetName('Insert Support 1')
        insert_support1.SetDisableTime(2, 's')

        insert_support2 = self.study.ImportWall(
            support2_stl_path,
            import_scale=self.particle_box_len,
            convert_yz=False)[0]
        insert_support2.SetName('Insert Support 2')
        insert_support2.SetDisableTime(2, 's')

        insert_support3 = self.study.ImportWall(
            support3_stl_path,
            import_scale=self.particle_box_len,
            convert_yz=True)[0]
        insert_support3.SetName('Insert Support 3')
        insert_support3.SetDisableTime(2, 's')
        insert_support3.SetTranslation([-self.particle_box_len/2, 0, 0])

        insert_support4 = self.study.ImportWall(
            support4_stl_path,
            import_scale=self.particle_box_len,
            convert_yz=True)[0]
        insert_support4.SetName('Insert Support 4')
        insert_support4.SetDisableTime(2, 's')
        insert_support4.SetTranslation([self.particle_box_len/2, 0, 0])

        # Insert inlet plane
        self.insert_inlet = self.study.ImportSurface(
            inlet_stl_path,
            import_scale=self.particle_box_len,
            convert_yz=False)[0]
        self.insert_inlet.SetName('Insert Inlet')

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

    def _set_particle_size(self):
        """
        Set the particle size and density
        """
        self.particle = self.study.CreateParticle()
        size_distr_lst = self.particle.GetSizeDistributionList()

        psd = size_distr_lst.New()
        psd.SetSize(self.p_radius, 'm')
        psd.SetCumulativePercentage(100)

    def _domain_settings(self):
        self.domain = self.study.GetDomainSettings()
        self.domain.SetDomainType('CARTESIAN')
        self.domain.SetUseBoundaryLimits(True)
        # Set periodic limits in X and Y directions
        # no periodic in Z-direction to allow compression
        self.domain.SetCartesianPeriodicDirections('XY')
        self.domain.SetPeriodicLimitsMaxCoordinates(
            [self.particle_box_len, self.particle_box_len, np.inf])
        self.domain.SetPeriodicLimitsMinCoordinates(
            [-self.particle_box_len, -self.particle_box_len, -np.inf])

    def _insertion_settings(self):
        fill_box_vol = self.particle_box_len**3
        particle_vol = (4/3) * np.pi * self.p_radius**3
        # 0.7 is a 'max' packing fraction
        n_particles = int(fill_box_vol * 0.7 / particle_vol)
        mass_particles = particle_vol * self.p_density * n_particles

        particle_inlet = self.study.CreateParticleInlet(
            self.insert_inlet, self.particle
        )
        input_property_lst = particle_inlet.GetInputPropertiesList()
        input_property_lst[0] .SetMassFlowRate(mass_particles, 'kg/s')

        particle_inlet.SetStartTime(0)
        particle_inlet.SetEndTime(1)
        particle_inlet.DisablePeriodic()

    def _compress_wall(self):
        frame_source = self.study.GetFrameSource()
        compr_motion_frame = frame_source.NewFrame()
        motions = compr_motion_frame.GetMotions()

        # Handling the free body motion
        freebody_motion = motions.New()
        freebody_motion.SetType('Free Body Translation')
        freebody = freebody_motion.GetTypeObject()
        freebody.SetFreeMotionDirection('Z')

        # Set the compression wall motion
        force_magnitude = self.compr_pressure * self.particle_box_len**2
        force_motion = motions.New()
        force_motion.SetType('Additional Force')
        add_force = force_motion.GetTypeObject()
        add_force.SetForceValue([0, 0, force_magnitude], 'N')
