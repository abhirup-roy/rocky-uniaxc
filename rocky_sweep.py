#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Abhirup Roy"
__email__ = "axr154@bham.ac.uk"
__status__ = "Development"

"""
This script generates multiple cases for Rocky DEM simulations using a template and a set of parameters.
It creates a directory for each case, populates the template with parameters, and generates the necessary mesh files.
It also creates a slurm sbatch script for each case and can automatically launch the job on a slurm cluster.
It uses the jinja2 library for templating and the gmsh library for mesh generation.

The script uses a JSON file to define the parameters for the simulations.
The parameters include properties of the particles, interactions, and experimental settings.
The script iterates over all combinations of parameters, creating a case directory for each combination.

Example usage:
    from rocky_sweep import make_cases
    make_cases(
        meshdir='meshes',
        json_path='params.json',
        template_dir='templates',
        autolaunch=True
    )
"""

import os
import json
import subprocess
from collections import OrderedDict
import itertools
from pprint import pprint

import jinja2
from compr_meshgen import create_meshes_efficiently



class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def iter_params(json_path: str = 'params.json'):
    """Iterate over all parameter combinations."""
    # Load the JSON file
    with open(json_path, 'r') as f_params:
        params = json.load(f_params, object_pairs_hook=OrderedDict)

    # Handle shape parameters - now it's an array of shape objects
    shape_list = params['shape']
    
    # Extract all possible values for each shape parameter
    shape_names = []
    vert_ars = []
    horiz_ars = []
    n_corners_list = []
    sq_degrees = []
    particle_paths = []
    
    for shape in shape_list:
        shape_names.append(shape.get('name', 'sphere'))
        vert_ars.append(shape.get('vert_ar', 1.0))
        horiz_ars.append(shape.get('horiz_ar', 1.0))
        n_corners_list.append(shape.get('n_corners', 6))
        sq_degrees.append(shape.get('sq_degrees', 1.0))  # Note: using 'sq_degrees' to match JSON
        particle_paths.append(shape.get('particle_path', ''))

    # Find all combinations of parameters
    param_combinations = itertools.product(
        params['particle_properties']['radius'],
        params['particle_properties']['density'],
        params['particle_properties']['poisson'],
        params['particle_properties']['youngmod'],
        params['inseractions']['pp']['fric_dyn'],
        params['inseractions']['pp']['fric_stat'],
        params['inseractions']['pp']['fric_rolling'],
        params['inseractions']['pp']['cor'],
        params['inseractions']['pw']['fric_dyn'],
        params['inseractions']['pw']['fric_stat'],
        params['inseractions']['pw']['cor'],
        params['experim_settings']['box_len'],
        params['experim_settings']['p_compress'],
        params['contact_model']['normal'],
        params['contact_model']['tangential'],
        params['contact_model']['rolling'],
        params['contact_model']['adhesion'],
        shape_list
    )
    return param_combinations


def slurm_sbatch(case_dir: str, autolaunch: bool = False, ncpus: int = 20):
    """Create a slurm sbatch script for each case.
    Change if needed.
    """
    # Define the sbatch script template
    # This is a simple template. You can modify it as needed.
    template = f"""#!/bin/bash
#SBATCH --job-name=uniaxc
#SBATCH --ntasks={ncpus}
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --time=5-0
#SBATCH --qos=bbdefault
#SBATCH --mail-type=ALL
#SBATCH --account=windowcr-astrazeneca-abhi


module purge; module load bluebear
module load bear-apps/2023a
module load ANSYS_Rocky/2024R2.0


#HOW TO USE:

# --simulate  		Processes from the beginning, in hidden mode, the project file name and location that follows in quotes.
# --ncpus=		Choose number
# --resume=		0 for off, 1 for on
# --use-gpu		0 for off, 1 for on
# --gpu-num		Choose number
# --script		Runs a script .py file, the name of which follows in quotes.
# --headless		Process from the beginning, in hidden mode, the script .py file name of which follows in quotes.

# Run the application

Rocky --script "script_uniax.py" --headless >> rocky.log

    """
    # Write the sbatch script to a file
    write_path = os.path.join(case_dir, 'runRocky.sh')

    #  Create the sbatch script in sweeping directory
    with open(write_path, 'w') as sbatch_file:
        sbatch_file.write(template)

    # Launch the sbatch script from each case directory
    if autolaunch:
        with cd(case_dir):
            try:
                result = subprocess.run(
                    ['sbatch', 'runRocky.sh'], check=True, capture_output=True, text=True)
                os.mkdir('plots')
                print(f"Job submitted successfully: {result.stdout}")
            except subprocess.CalledProcessError as e:
                print(f"Error submitting job: {e.stderr}")


def make_cases(
    sweep_name: str,
    meshdir: str = 'meshes',
    json_path: str = 'params.json',
    template_dir: str ='templates',
    autolaunch: bool = True,
    ncpus: int = 20
):
    """Generate and launch cases with improved performance."""
    # Ensure the template directory exists
    template_dir = os.path.abspath(template_dir)
    if not os.path.exists(template_dir):
        raise FileNotFoundError(f"Directory {template_dir} does not exist.")

    # Load template once
    rocky_templ_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(f'{template_dir}')
    )
    rocky_template = rocky_templ_env.get_template('template_uniax.py')

    # Get all parameter combinations
    all_params = list(iter_params(json_path))
    total_cases = len(all_params)
    print(f"Setting up {total_cases} cases...")

    # Create the sweep directory
    os.makedirs(sweep_name, exist_ok=True)

    # Create directories for all cases first (parallel processing preparation)
    case_dirs = []
    for i in range(total_cases):
        case_dir = os.path.join(sweep_name, f'case_{i}')
        os.makedirs(case_dir, exist_ok=True)
        os.makedirs(os.path.join(case_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(case_dir, meshdir), exist_ok=True)
        case_dirs.append(case_dir)

    # Generate meshes only once per unique size
    # This is a major optimization - don't recreate identical meshes
    unique_sizes = set([params[11]
                       for params in all_params])  # Box length parameter
    size_to_mesh_dir = {}

    print(f"Generating meshes for {len(unique_sizes)} unique sizes...")
    for size in unique_sizes:
        # Create a shared mesh directory for this size
        shared_mesh_dir = os.path.join(sweep_name, f'meshes_{size}')
        os.makedirs(shared_mesh_dir, exist_ok=True)

        # Generate meshes only once for each unique size
        create_meshes_efficiently(size, meshsize=0.01, out_dir=shared_mesh_dir)

        size_to_mesh_dir[size] = shared_mesh_dir

    # Write scripts and prepare to launch
    print("Generating scripts and preparing jobs...")
    for i, params in enumerate(all_params):
        case_dir = case_dirs[i]
        box_size = params[11]

        print(params)

        # Prepare script context
        script_contxt = {
            'RADIUS_P': params[0],
            'DENSITY_P': params[1],
            'POISSON_P': params[2],
            'YOUNGMOD_P': params[3],
            'DYNAMIC_FRICTION_PP': params[4],
            'STATIC_FRICTION_PP': params[5],
            'COR_PP': params[7],
            'DYNAMIC_FRICTION_PW': params[8],
            'STATIC_FRICTION_PW': params[9],
            'COR_PW': params[10],
            'L_BOX': params[11],
            'P_COMPRESS': params[12],
            'NORMAL_MODEL': params[13],
            'TANG_MODEL': params[14],
            'ROLLING_MODEL': params[15],
            'ADH_MODEL': params[16],
            'SHAPE': params[-1].get('name', 'sphere'),  # Use the name from the shape object
            'VERT_AR': params[-1].get('vert_ar', 0.5),
            'HORIZ_AR': params[-1].get('horiz_ar', 1.0),
            'N_CORNERS': params[-1].get('n_corners', 8),
            'SQ_DEGREE': params[-1].get('sq_degree', 2.0),
            'PARTICLE_PATH': params[-1].get('particle_path', ''),
            'SMOOTHNESS': params[-1].get('smoothness', 0.5),
            'MESH_DIR': str(meshdir),
        }

        if params[15] != '"none"':
            script_contxt['ROLLING_FRICTION'] = params[6]

        print(params)

        # Render template and write script
        rendered_content = rocky_template.render(script_contxt)
        script_path = os.path.join(case_dir, 'script_uniax.py')

        with open(script_path, 'w') as script_file:
            script_file.write(rendered_content)

        # Log case information
        print(f"Case {i}/{total_cases} prepared")

        # Create SLURM script
        slurm_sbatch(case_dir, autolaunch=False, ncpus=ncpus)  # Don't launch yet

    # Launch all cases at once if requested
    if autolaunch:
        print("Launching all cases...")
        for i, case_dir in enumerate(case_dirs):
            print(f"Launching case {i}/{total_cases}...")
            # Use subprocess to launch in the background
            with cd(case_dir):
                try:
                    result = subprocess.run(
                        ['sbatch', 'runRocky.sh'], check=True, capture_output=True, text=True)
                    print(f"Job submitted: {result.stdout.strip()}")
                except subprocess.CalledProcessError as e:
                    print(f"Error submitting job: {e.stderr}")

    print(f"All {total_cases} cases prepared and launched.")
    print(f"Exiting launcher script now")


if __name__ == "__main__":
    make_cases(
        sweep_name='polyh_corners_highcpu',
        json_path='json/polyh_corners.json',
        autolaunch=True,
        ncpus=64
    )
