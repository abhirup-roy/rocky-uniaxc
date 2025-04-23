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

from compr_meshgen import create_particlebox, create_compr_walls, create_insert

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def iter_params(json_path: str='params.json'):
    """Iterate over all parameter combinations."""
    # Load the JSON file
    with open(json_path, 'r') as f_params:
        params = json.load(f_params, object_pairs_hook=OrderedDict)

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
        params['inseractions']['pw']['fric_rolling'],
        params['inseractions']['pw']['cor'],
        params['experim_settings']['box_len'],
        params['experim_settings']['p_compress'],
        params['contact_model']['normal'],
        params['contact_model']['tangential'],
        params['contact_model']['rolling'],
        params['contact_model']['adhesion']
    )

    pprint(params)
    return param_combinations


def slurm_sbatch(case_dir: str, autolaunch: bool = False):
    """Create a slurm sbatch script for each case.
    Change if needed.
    """
    # Define the sbatch script template
    # This is a simple template. You can modify it as needed.
    template = """#!/bin/bash
#SBATCH --job-name=uniaxc
#SBATCH --ntasks=20
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
                result = subprocess.run(['sbatch', 'runRocky.sh'], check=True, capture_output=True, text=True)
                os.mkdir('plots')
                print(f"Job submitted successfully: {result.stdout}")
            except subprocess.CalledProcessError as e:
                print(f"Error submitting job: {e.stderr}")

def make_cases(
        meshdir: str = 'meshes',
        json_path: str = 'params.json',
        template_dir = 'templates',
        autolaunch = True
        ):

    # Get ensuring the template directory exists
    template_dir = os.path.abspath(template_dir)
    if not os.path.exists(template_dir):
        raise FileNotFoundError(f"Directory {template_dir} does not exist.")

    # Populate template with parameters
    rocky_templ_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(f'{template_dir}')
    )
    rocky_template = rocky_templ_env.get_template('template_uniax.py')

    for i, params in enumerate(iter_params(json_path)):
        # Create a directory for each case
        # Include a plots directory for each case
        case_dir = f"case_{i}"
        os.makedirs(case_dir, exist_ok=True)

        # Generate the mesh files
        with cd(case_dir):
            if not os.path.exists(meshdir):
                os.makedirs(meshdir)
            create_particlebox(params[12], meshsize=0.001, gui=False)
            create_compr_walls(params[12], meshsize=0.001, gui=False)
            create_insert(params[12], meshsize=0.001, gui=False)

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
            'COR_PW': params[11],
            'L_BOX': params[12],
            'P_COMPRESS': params[13],
            'NORMAL_MODEL': params[14],
            'TANG_MODEL': params[15],
            'ROLLING_MODEL': params[16],
            'ADH_MODEL': params[17],
            'MESH_DIR': str(meshdir),
        }

        if params[16] != '"none"':
            script_contxt['ROLLING_FRICTION_PP'] = params[6]
            script_contxt['ROLLING_FRICTION_PW'] = params[10]

        # Render the template with the parameters
        rendered_content = rocky_template.render(script_contxt)
        # Write the rendered content to a file
        script_path = os.path.join(case_dir, 'script_uniax.py')

        with open(script_path, 'w') as script_file:
            script_file.write(rendered_content)

        print(f"Case {i}:")
        pprint(f"Parameters: {params}")
        print(f"Case {i}:")
        pprint(f"Parameters: {params}")
        print(f"Script written to {script_path}")

        print(f"Launching case {i}...")
        slurm_sbatch(case_dir, autolaunch=autolaunch)

    print(f"Exiting launcher script now") 

if __name__ == "__main__":
    make_cases(autolaunch=True)