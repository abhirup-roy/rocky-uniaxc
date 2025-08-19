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

import numpy as np
import pandas as pd
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


def iter_params(json_path: str):
    """
    Iterate over all parameter combinations.

    Args:
        json_path: Path to json config for sweep
    """
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


def slurm_sbatch(case_dir: str, loc: str, autolaunch: bool = False, 
    custom_msg: str = None, ncpus: int = None):
    """Create a slurm sbatch script for each case.
    Change if needed.
    """

    if loc == 'bb-cpu' and not ncpus:
        ncpus = 20

    # Define the sbatch script template
    # This is a simple template. You can modify it as needed.

    # For UoB BlueBear use
    if loc == 'bb-cpu':
        template = f"""#!/bin/bash
#SBATCH --job-name=uniaxc
#SBATCH --ntasks={ncpus}
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --time=5-0
#SBATCH --qos=bbdefault
#SBATCH --mail-type=ALL
#SBATCH --account=windowcr-astrazeneca-abhi

set -e

module purge; module load bluebear
module load bear-apps/2023a
module load ANSYS_Rocky/2024R2.0

Rocky --script "script_uniax.py" --headless >> rocky.log
    """

    # For AZ SCP use
    elif loc == "az-gpu":
        template="""#!/bin/sh
#SBATCH -L uniaxc
#SBATCH --ntasks=1
#SBATCH --time=5-0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=1
#SBATCH -p gpu
#SBATCH -L ansys:1

set -e

ml rocky/24.2.0

Rocky --script "script_uniax.py" --headless >> rocky.log

    """
    elif loc == 'custom':
        if custom_msg and custom_msg.startswith('#!/bin/bash'):
            template = custom_msg
        else:
            raise ValueError('Invalid custom message provided')
        
    else:
        raise ValueError('Only')
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
        template_dir='templates',
        autolaunch=True,
        loc: str = 'bb-cpu',
        custom_sh: str = None,
        target: str = 'CPU',
        ncpus: int = None
):
    """
    Generate and launch sweep cases

    Args:
        sweep_name: A string for the title of the sweep being carried out
        json_path: A path to the json config for the sweep or simulation
        template_dir: A path to the script templates
        autolaunch: Whether to automatically launch scripts
        loc: Specify cluster script to use. Currently only works with
            'az-gpu', 'bb-cpu', 'custom'. N.B. If using custom, specify
            the `custom_sh` arg
        custom_sh: A custom SLURM script to run simulations. Only needed if
            using `loc=custom`
    """
    # Ensure the template directory exists
    template_dir = os.path.abspath(template_dir)
    if not os.path.exists(template_dir):
        raise FileNotFoundError(f"Directory {template_dir} does not exist.")
    
    target = target.upper()
    if target not in ['CPU', 'GPU', 'MULTI_GPU']:
        raise ValueError("Select from 'CPU', 'GPU', 'MULTI_GPU'")
    elif target == 'MULTI_GPU':
        raise NotImplementedError('Multi GPU use not validated yet')
    
    if (loc == 'bb-cpu' and target == 'GPU') or (loc == 'az-gpu' and target == 'CPU'):
        raise ValueError(f'{target} is not valid for location {loc}')
    target = '"' + target + '"'
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
            'XPU': target
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
        slurm_sbatch(case_dir, loc=loc, autolaunch=False, custom_msg=custom_sh, ncpus=ncpus)  # Don't launch yet

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


def iter_ofat(json_path: str, ofat_values: dict[str, list|str], n_points:int):
    """
    Iterate over all parameter combinations.

    Args:
        json_path: Path to json config for sweep
        ofat_values: A dictionary containing the OFAT parameters, test range.
            Accepts the following keys: 'parameters', 'test_range', and 'hold_values'.
    """
    # Load the JSON file
    with open(json_path, 'r') as f_params:
        params = json.load(f_params, object_pairs_hook=OrderedDict)

    # Handle shape parameters - now it's an array of shape objects
    shape = params['shape']
    if isinstance(shape, list):
        raise ValueError("Shape parameters should be a single object, not a list.")

    # Manual check for parameter types
    # Base case should NOT be a list
    for p in (
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
        shape
    ):
        if isinstance(p, list):
            raise ValueError(f"Parameter values should not be lists. Use a single value for {p}.")
    
    # Check if ofat_values is a dictionary and contains the required keys
    if not isinstance(ofat_values, dict):
        raise ValueError("OFAT values must be provided as a dictionary.")

    
    ofat_dict_check = set(ofat_values.keys()) == set(['parameters', 'test_range', 'hold_values'])
    if not ofat_dict_check:
        raise ValueError("OFAT values must contain 'parameters', 'test_range', and 'hold_values' keys.")

    ofat_base_valid = {
        'radius': params['particle_properties']['radius'],  # float
        'density': params['particle_properties']['density'],  # float
        'poisson': params['particle_properties']['poisson'],  # float
        'youngmod': params['particle_properties']['youngmod'],  # float
        'fric_dyn_pp': params['inseractions']['pp']['fric_dyn'],  # float
        'fric_stat_pp': params['inseractions']['pp']['fric_stat'],  # float
        'fric_rolling_pp': params['inseractions']['pp']['fric_rolling'],  # float
        'cor_pp': params['inseractions']['pp']['cor'],  # float
        'fric_dyn_pw': params['inseractions']['pw']['fric_dyn'],  # float
        'fric_stat_pw': params['inseractions']['pw']['fric_stat'],  # float
        'cor_pw': params['inseractions']['pw']['cor'],  # float
        'box_len': params['experim_settings']['box_len'],  # float
        'p_compress': params['experim_settings']['p_compress'],  # float
        'normal': params['contact_model']['normal'],  # float
        'tangential': params['contact_model']['tangential'],  # float
        'rolling': params['contact_model']['rolling'],  # float
        'adhesion': params['contact_model']['adhesion'],  # float
        'shape': params['shape']['name'],  # string
        'vert_ar': params['shape']['vert_ar'],  # float
        'horiz_ar': params['shape']['horiz_ar'],  # float
        'n_corners': params['shape']['n_corners'],  # int
        'sq_degree': params['shape']['sq_degree'], # float
    }


    if not set(ofat_values['parameters']).issubset(set(ofat_base_valid.keys())):
        raise ValueError(f"Invalid OFAT parameters. Allowed parameters are: {list(ofat_base_valid.keys())}")

    range_valid = {
        'fric_dyn_pp': (0, 1),
        'fric_stat_pp': (0, 1),
        'fric_rolling_pp': (0, 1),
        'cor_pp': (0, 1),
        'fric_dyn_pw': (0, 1),
        'fric_stat_pw': (0, 1),
        'cor_pw': (0, 1),
        'box_len': (0, None),
        'p_compress': (0, 1),
        'normal': (0, 1),
        'tangential': (0, 1),
        'rolling': (0, 1),
        'adhesion': (0, 1),
        'vert_ar': (0, None),
        'horiz_ar': (0, None),
        'n_corners': (0, None),
        'sq_degree': (2, None)
    }

    for k in ofat_values['parameters']:
        if k not in range_valid:
            raise ValueError(f"Parameter '{k}' is not recognized for range validation.")
        lb, ub = range_valid[k]
        ub = ub if ub is not None else float('inf')
        if k not in ofat_base_valid:
            raise ValueError(f"Parameter '{k}' is not in the base parameters.")
        if not (lb <= ofat_base_valid[k] <= ub):
            raise ValueError(f"Base parameter '{k}' with value {ofat_base_valid[k]} is out of range ({lb}, {ub}).")

        param_idx = ofat_values['parameters'].index(k)
        test_range = ofat_values['test_range'][param_idx]
        hold_value = ofat_values['hold_values'][param_idx]
        
        if hold_value not in ['h', 'l', 'm']:
            raise ValueError(f"Hold value '{hold_value}' for parameter '{k}' is not valid. Use 'h', 'l', or 'm'.")
        
        lb_i, ub_i = test_range
        if lb_i >= ub_i:
            raise ValueError(f"Invalid test range for parameter '{k}': ({lb_i}, {ub_i})")
        elif not (lb <= lb_i <= ub and lb <= ub_i <= ub):
            raise ValueError(f"Test range for parameter '{k}' with values ({lb_i}, {ub_i}) is out of bounds ({lb}, {ub}).")

    if len(ofat_values['parameters']) != len(ofat_values['test_range']) or \
       len(ofat_values['hold_values']) != len(ofat_values['parameters']):
        raise ValueError("Mismatched lengths in OFAT values.")
    
    levels = {}
    for i, rng in enumerate(ofat_values['test_range']):
        lb, ub = rng
        if lb >= ub:
            raise ValueError(f"Invalid range for parameter '{ofat_values['parameters'][i]}': ({lb}, {ub})")

        dtype = int if ofat_values['parameters'][i] == 'n_corners' else float
        levels_i = np.linspace(lb, ub, n_points, dtype=dtype)
        if ofat_values['hold_values'][i] == 'h':
            hold_i = levels_i[-1]
        elif ofat_values['hold_values'][i] == 'l':
            hold_i = levels_i[0]
        elif ofat_values['hold_values'][i] == 'm':
            if levels_i.size % 2 == 0:
                hold_i = levels_i[levels_i.size // 2 - 1:levels_i.size // 2 + 1]
            else:
                hold_i = levels_i[levels_i.size // 2]
        else:
            raise ValueError(f"Invalid hold value for parameter '{ofat_values['parameters'][i]}':\
                              {ofat_values['hold_values'][i]}. Select from 'h', 'l', 'm'.")
        levels[ofat_values['parameters'][i]] = {
            'levels': levels_i,
            'hold': hold_i
        }

    baseline = {param: v['hold'] for param, v in levels.items()}
    experiments = [baseline.copy()]

    for factor, v in levels.items():
        for level in v['levels']:
            if level != baseline[factor]:
                experiment = baseline.copy()
                experiment[factor] = level
                experiments.append(experiment)
    
    experiments_df = pd.DataFrame(experiments)

    ofat_vars = set(experiments_df.columns)
    base_vars = set(ofat_base_valid.keys())

    keys_to_drop = base_vars & ofat_vars
    for k in keys_to_drop:
        del ofat_base_valid[k]

    return experiments_df, ofat_base_valid

def launch_ofat(sweep_name: str, autolaunch: bool, json_path:str,
                ofat_values: dict[str, list|str], loc: str = 'bb-cpu',
                target: str = 'CPU', n_points: int = 10,
                ncpus: int = None, **kwargs):

    custom_sh = kwargs.get('custom_sh', None)
    template_dir = kwargs.get('template_dir', 'templates')
    template_dir = os.path.abspath(template_dir)
    if not os.path.exists(template_dir):
        raise FileNotFoundError(f"Directory {template_dir} does not exist.")
    
    target = target.upper()
    if target not in ['CPU', 'GPU', 'MULTI_GPU']:
        raise ValueError("Select from 'CPU', 'GPU', 'MULTI_GPU'")
    elif target == 'MULTI_GPU':
        raise NotImplementedError('Multi GPU use not validated yet')
    
    if (loc == 'bb-cpu' and target == 'GPU') or (loc == 'az-gpu' and target == 'CPU'):
        raise ValueError(f'{target} is not valid for location {loc}')
    target = '"' + target + '"'
    # Load template once
    rocky_templ_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(f'{template_dir}')
    )
    rocky_template = rocky_templ_env.get_template('template_uniax.py')

    experiments_df, base_dict = iter_ofat(
        json_path=json_path, ofat_values=ofat_values, n_points=n_points
    )

    total_cases = len(experiments_df)
    vars = experiments_df.columns.tolist()

    # Generate all cases
    all_params = []

    for _, row in experiments_df.iterrows():
        exp_dict = {}
        for i in range(len(vars)):
            exp_dict[vars[i]] = row[vars[i]]
        exp_dict.update(base_dict)
        all_params.append(exp_dict)

    # Create the sweep directory
    os.makedirs(sweep_name, exist_ok=True)

    # Create directories for all cases first (parallel processing preparation)
    case_dirs = []
    for i in range(total_cases):
        case_dir = os.path.join(sweep_name, f'case_{i}')
        os.makedirs(case_dir, exist_ok=True)
        os.makedirs(os.path.join(case_dir, 'plots'), exist_ok=True)
        case_dirs.append(case_dir)

    if 'box_len' in experiments_df.columns:
        unique_sizes = set(experiments_df['box_len'])
    elif 'box_len' in base_dict.keys():
        unique_sizes = [base_dict['box_len']]
    else:
        raise ValueError("No box length parameter found in experiments or base dictionary. "
                         "Debugging required")
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

        # Prepare script context
        script_contxt = {
            'RADIUS_P': params['radius'],
            'DENSITY_P': params['density'],
            'POISSON_P': params['poisson'],
            'YOUNGMOD_P': params['youngmod'],
            'DYNAMIC_FRICTION_PP': params['fric_dyn_pp'],
            'STATIC_FRICTION_PP': params['fric_stat_pp'],
            'COR_PP': params['cor_pp'],
            'DYNAMIC_FRICTION_PW': params['fric_dyn_pw'],
            'STATIC_FRICTION_PW': params['fric_stat_pp'],
            'COR_PW': params['cor_pw'],
            'L_BOX': params['box_len'],
            'P_COMPRESS': params['p_compress'],
            'NORMAL_MODEL': params['normal'],
            'TANG_MODEL': params['tangential'],
            'ROLLING_MODEL': params['rolling'],
            'ADH_MODEL': params['adhesion'],
            'SHAPE': params['shape'],
            'VERT_AR': params.get('vert_ar'),
            'HORIZ_AR': params.get('horiz_ar'),
            'N_CORNERS': params.get('n_corners'),
            'SQ_DEGREE': params.get('sq_degree'),
            'PARTICLE_PATH': params.get('particle_path'),
            'SMOOTHNESS': params.get('smoothness', 0.5),
            'XPU': target
        }

        if params['rolling'] != 'none':
            script_contxt['ROLLING_FRICTION'] = params['rolling']
        else:
            script_contxt['ROLLING_FRICTION'] = None

        rendered_content = rocky_template.render(script_contxt)
        script_path = os.path.join(case_dir, 'script_uniax.py')

        with open(script_path, 'w') as script_file:
            script_file.write(rendered_content)

        # Log case information
        print(f"Case {i}/{total_cases} prepared")

        # Create SLURM script
        slurm_sbatch(case_dir, loc=loc, autolaunch=False, custom_msg=custom_sh, ncpus=ncpus)  # Don't launch yet

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
                        ['sbatch', 'runRocky.sh'], check=True, capture_output=True, text=True)
                    print(f"Job submitted: {result.stdout.strip()}")
                except subprocess.CalledProcessError as e:
                    print(f"Error submitting job: {e.stderr}")


    print(f"All {total_cases} cases prepared and launched.")
    print(f"Exiting launcher script now")

if __name__ == "__main__":

    """Example of a regular sweep"""
    # make_cases(
    #     sweep_name='reg_sweep_example',
    #     json_path='json/swesp_reg.json',
    #     autolaunch=True,
    #     loc='az-gpu',
    #     target='GPU'
    # )

    """Example of an OFAT sweep"""
    # launch_ofat(
    #     'ofat_example',
    #     autolaunch=True,
    #     json_path='json/ofat_base.json',
    #     ofat_values={
    #         'parameters':['n_corners', 'sq_degree'],
    #         'test_range':[(5, 50), (2.0, 10.0)],
    #         'hold_values': ['m', 'm']
    #     },
    #     n_points=5,
    #     loc='bb-cpu',
    #     target='CPU',
    #     ncpus=20
    # )
