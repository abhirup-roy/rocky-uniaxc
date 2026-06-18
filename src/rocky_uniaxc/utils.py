import os
import pathlib
import subprocess
from typing import Optional


class cd:
    """Context manager for temporarily changing the current working directory.

    Usage::

        with cd("/tmp"):
            # working directory is now /tmp
            ...
        # original directory is restored
    """

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def slurm_sbatch(
    case_dir: str | pathlib.Path,
    loc: str,
    autolaunch: bool = False,
    custom_msg: Optional[str] = None,
    ncpus: Optional[int] = None,
    ngpus: int = 1,
    run_days: int = 12,
):
    """Create a SLURM sbatch script for a simulation case.

    Generates a ``runRocky.sh`` script in the case directory using a template
    appropriate for the specified cluster location, then optionally launches it
    via ``sbatch``.

    Args:
        case_dir: Directory of the case for which to create the sbatch script.
        loc: Cluster location determining the script template. Accepted values
            are ``"bb-cpu"`` (BlueBear CPU), ``"bb-gpu"`` (BlueBear GPU),
            ``"az-gpu"`` (Azure GPU), and ``"custom"``.
        autolaunch: If ``True``, automatically submit the script via
            ``sbatch`` after writing it.
        custom_msg: Custom sbatch script template. Required when
            ``loc="custom"``. Must start with the shebang line
            (``#!/bin/bash``).
        ncpus: Number of CPUs to request. Only applicable when
            ``loc="bb-cpu"``. Defaults to 20.
        ngpus: Number of GPUs to request. Only applicable when
            ``loc="bb-gpu"`` or ``loc="az-gpu"``. Defaults to 1.
        run_days: Number of days to request for the job runtime. Defaults
            to 12.

    Raises:
        ValueError: If ``loc`` is not one of the supported locations, or if
            ``loc="custom"`` and ``custom_msg`` does not start with
            ``#!/bin/bash``.
    """

    if loc == "bb-cpu" and not ncpus:
        ncpus = 20

    # Define the sbatch script template
    # This is a simple template. You can modify it as needed.

    # For UoB BlueBear use
    if loc == "bb-cpu":
        template = f"""#!/bin/bash
#SBATCH --job-name=uniaxc
#SBATCH --ntasks={ncpus}
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --time={run_days}-0
#SBATCH --qos=bbdefault
#SBATCH --mail-type=ALL
#SBATCH --account=windowcr-astrazeneca-abhi

set -e

module purge; module load bluebear
module load bear-apps/2024a
module load ANSYS_Rocky/2025R2

Rocky --script "script_uniax.py" --headless >> rocky.log
    """

    # For AZ SCP use
    elif loc == "az-gpu":
        template = f"""#!/bin/sh
#SBATCH --job-name=uniaxc
#SBATCH --ntasks=1
#SBATCH --time={run_days}-0
#SBATCH --gres=gpu:{ngpus}
#SBATCH --cpus-per-gpu=1
#SBATCH -p gpu

set -e

ml rocky/25.2.0

Rocky --script "script_uniax.py" --headless >> rocky.log

    """

    elif loc == "bb-gpu":
        template = f"""#!/bin/bash
#SBATCH --job-name=uniaxc
#SBATCH --ntasks=1
#SBATCH --time={run_days}-0
#SBATCH --gres=gpu:{ngpus}
#SBATCH --qos=bbgpu
#SBATCH --account=windowcr-astrazeneca-abhi
#SBATCH --gres=gpu:1

set -e

module purge; module load bluebear
module load bear-apps/2024a
module load ANSYS_Rocky/2025R2

Rocky --script "script_uniax.py" --headless >> rocky.log
    """
    elif loc == "custom":
        if custom_msg and custom_msg.startswith("#!/bin/bash"):
            template = custom_msg
        else:
            raise ValueError("Invalid custom message provided")

    else:
        raise ValueError(
            "Only 'bb-cpu', 'bb-gpu', 'az-gpu' and 'custom' locations are supported"
            f" but got '{loc}'"
        )
    # Write the sbatch script to a file
    write_path = (pathlib.Path(case_dir) / "runRocky.sh").resolve()

    #  Create the sbatch script in sweeping directory
    with open(write_path, "w") as sbatch_file:
        sbatch_file.write(template)

    # Launch the sbatch script from each case directory
    if autolaunch:
        with cd(case_dir):
            try:
                result = subprocess.run(
                    ["sbatch", "runRocky.sh"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                os.mkdir("plots")
                print(f"Job submitted successfully: {result.stdout}")
            except subprocess.CalledProcessError as e:
                print(f"Error submitting job: {e.stderr}")
