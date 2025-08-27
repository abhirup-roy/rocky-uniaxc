import os
import subprocess


class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

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
#SBATCH --job-name=uniaxc
#SBATCH --ntasks=1
#SBATCH --time=5-0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=1
#SBATCH -p gpu

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
