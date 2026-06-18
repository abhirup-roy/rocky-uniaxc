"""Design of Experiments (DOE) sub-package for rocky_uniaxc.

Provides job-launching utilities and shared constants for sweep and OFAT
experiment workflows.
"""

import pathlib
import subprocess
from tqdm import tqdm
from ..utils import cd


def _tqdm_launch(case_dirs, total_cases):
    """Launch simulation jobs via ``sbatch`` with a tqdm progress bar.

    Args:
        case_dirs: Iterable of case directory paths.
        total_cases: Total number of cases (for the progress bar).

    Side Effects:
        Submits SLURM jobs and prints a summary of any failures.
    """
    failed_jobs = []
    for i, case_dir in tqdm(
        enumerate(case_dirs), total=total_cases, desc="Submitting Jobs", unit="case"
    ):
        # Use subprocess to launch in the background
        with cd(case_dir):
            try:
                subprocess.run(
                    ["sbatch", "runRocky.sh"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                # Use tqdm.write to print errors without breaking the progress bar layout
                err_msg = e.stderr.strip() if e.stderr else "Unknown error"
                tqdm.write(f"Failed to launch case {i}: {err_msg}")
                failed_jobs.append(i)
            except Exception as e:
                tqdm.write(f"Unexpected error in case {i}: {e}")
                failed_jobs.append(i)

    if failed_jobs:
        print(
            f"\nCompleted with errors. {len(failed_jobs)} jobs failed to launch (Case indices: {failed_jobs})."
        )
    else:
        print("\nSuccess: All jobs launched.")

    print(f"All {total_cases} cases prepared and processed.")
    print("Exiting launcher script now")


shapes_module_path = str(
    (pathlib.Path(__file__).parent.parent / "particles_shapes.py").resolve()
)
