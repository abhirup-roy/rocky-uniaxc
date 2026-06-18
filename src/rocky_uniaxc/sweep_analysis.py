"""Post-processing and analysis tools for uniaxial compression sweep results.

Provides utilities for loading, filtering, and exporting simulation data from
SQLite databases, as well as functions for identifying faulty runs and
repeating sweeps.
"""

import glob
import shutil
import re
import os
import sqlite3
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
from .utils import cd

PWD = os.getcwd()


def load_data(sweep_name: str) -> pd.DataFrame:
    """Load the results DataFrame from the ``results.db`` SQLite database.

    Args:
        sweep_name: Name of the sweep directory containing ``results.db``.

    Returns:
        A :class:`~pandas.DataFrame` with the query results, indexed by
        ``id``.

    Raises:
        FileNotFoundError: If the database file does not exist.
    """
    db_path = os.path.join(PWD, sweep_name, "results.db")

    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"Results db not found at {db_path}")

    read_table_sql = "SELECT * FROM results;"
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(read_table_sql, conn, index_col="id")

    return df


def _write_df_file(df: pd.DataFrame, name: str, filetype: str, s_dir: str):
    """Write a DataFrame to a file in the specified directory and format.

    Args:
        df: The DataFrame to write.
        name: Base filename (without extension).
        filetype: Output format — ``"csv"``, ``"parquet"``, ``"feather"``,
            or ``"excel"``.
        s_dir: Directory to write the file into.
    """
    if filetype == "csv":
        df.to_csv(os.path.join(s_dir, f"{name}.csv"))
    elif filetype == "parquet":
        df.to_parquet(os.path.join(s_dir, f"{name}.parquet.gzip"), compression="gzip")
    elif filetype == "feather":
        df.to_feather(os.path.join(s_dir, f"{name}.feather"))
    elif filetype == "excel":
        df.to_excel(os.path.join(s_dir, f"{name}.xlsx"))


def dump_results(
    sweep_name: str,
    filetype: str = "csv",
    outputs_dir: str = "pyoutputs",
    minimal: bool = False,
):
    """Export the results DataFrame to a file in the outputs directory.

    Args:
        sweep_name: Name of the sweep directory containing ``results.db``.
        filetype: Output format — ``"csv"``, ``"parquet"``, or
            ``"feather"``. Defaults to ``"csv"``.
        outputs_dir: Subdirectory within the sweep directory to save the
            file. Defaults to ``"pyoutputs"``.
        minimal: If ``True``, drop columns with only one unique value.
    """
    outputs_dir_path = os.path.join(PWD, sweep_name, outputs_dir)
    if not os.path.isdir(outputs_dir):
        os.makedirs(outputs_dir_path, exist_ok=True)

    df = load_data(sweep_name=sweep_name).set_index("case_n").sort_index()
    if minimal:
        df = df.loc[:, df.nunique(dropna=True) > 1]

    _write_df_file(df=df, name="results", filetype=filetype, s_dir=outputs_dir_path)
    if filetype == "csv":
        dump_path = os.path.join(outputs_dir_path, "results.csv")
        df.to_csv(dump_path)
    elif filetype == "parquet":
        dump_path = os.path.join(outputs_dir_path, "results.parquet.gzip")
        df.to_parquet(dump_path, compression="gzip")
    elif filetype == "feather":
        dump_path = os.path.join(outputs_dir_path, "results.feather")
        df.to_feather(dump_path)


def find_faulty_runs(sweep_name: str, dump: bool = False):
    """Scan case directories for potentially faulty simulation results.

    Checks for Hausner ratios outside the typical range of 1–3 and scans
    SLURM output files for warnings about particle loss.  Results are
    printed and optionally dumped to a text file.

    Args:
        sweep_name: Name of the sweep directory to scan.
        dump: If ``True``, write the list of faulty cases to
            ``faulty_cases.txt`` in the sweep directory.
    """
    sweep_dir = os.path.join(PWD, sweep_name)
    # Get all subdirectories that start with 'case_'
    case_dirs = [
        entry.name
        for entry in os.scandir(sweep_dir)
        if entry.is_dir() and entry.name.startswith("case_")
    ]

    faulty_cases = {}
    for case_dir in case_dirs:
        case_path = os.path.join(sweep_dir, case_dir)
        results_file = os.path.join(case_path, "results.csv")

        if not os.path.isfile(results_file):
            continue
        else:
            ser = pd.read_csv(results_file).squeeze()
            if ser["hausner_ratio"] < 1 or ser["hausner_ratio"] > 3:
                faulty_cases[case_dir] = ser["hausner_ratio"]

        output_file = glob.glob(os.path.join(case_path, "slurm-*.out"))
        if output_file:
            with open(output_file[0], "r") as f:
                content = f.read()
                if (
                    "RuntimeWarning: Particles were lost during the simulation"
                    in content
                ):
                    particles_init = int(
                        re.search(r"Initial particle count: (\d+)", content).group(1)
                    )
                    particles_final = int(
                        re.search(r"Final particle count: (\d+)", content).group(1)
                    )
                    print(
                        f"Warning: Particles lost in {case_dir}: {particles_init} -> {particles_final} ({particles_final - particles_init})"
                    )

    print(f"Faulty cases in sweep '{sweep_name}':")

    if faulty_cases:
        for case, hr_value in faulty_cases.items():
            print(f"  - {case}:   HR = {hr_value:.3f}")
    else:
        print("  No faulty cases found.")
    if dump:
        with open(os.path.join(sweep_dir, "faulty_cases.txt"), "w") as f:
            for case in faulty_cases.items():
                f.write(f"{case}\n")
            print(
                f"  Faulty cases dumped to {os.path.join(sweep_dir, 'faulty_cases.txt')}"
            )


def dump_results_backup(sweep_name: str, filetype: str = "csv", minimal: bool = False):
    """Fallback export that combines individual ``results.csv`` files.

    Reads the per-case ``results.csv`` files and concatenates them into a
    single DataFrame, then writes the result to the sweep directory.

    Args:
        sweep_name: Name of the sweep directory containing ``case_*``
            subdirectories.
        filetype: Output format — ``"csv"``, ``"parquet"``, ``"feather"``,
            or ``"excel"``. Defaults to ``"csv"``.
        minimal: If ``True``, drop columns with only one unique value.

    Raises:
        NotADirectoryError: If the sweep directory does not exist.
    """
    sweep_dir = os.path.join(PWD, sweep_name)
    if not os.path.isdir(sweep_dir):
        raise NotADirectoryError(f"Sweep directory not found at {sweep_dir}")

    case_dirs = [
        entry.name
        for entry in os.scandir(sweep_dir)
        if entry.is_dir() and entry.name.startswith("case_")
    ]

    all_df = pd.DataFrame()
    for case_dir in case_dirs:
        case_path = os.path.join(sweep_dir, case_dir)
        results_file = os.path.join(case_path, "results.csv")
        if not os.path.isfile(results_file):
            continue
        df = pd.read_csv(results_file).set_index("case_n")
        all_df = pd.concat([all_df, df])
    all_df.sort_index(inplace=True)

    if minimal:
        all_df = all_df.loc[:, all_df.nunique(dropna=True) > 1]

    _write_df_file(all_df, name="all_results", filetype=filetype, s_dir=sweep_dir)


def _insert_line_in_file(file_path: str, match_str: str, new_line: str):
    """Insert a line into a file below a matching line, preserving indentation.

    Args:
        file_path: Path to the file to modify.
        match_str: Substring to search for in existing lines.
        new_line: Text to insert below the matched line.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    modified = False
    with open(file_path, "w") as f:
        for line in lines:
            f.write(line)
            if match_str in line:
                # Capture indentation (leading whitespace)
                indent = line[: len(line) - len(line.lstrip())]
                f.write(f"{indent}{new_line}\n")
                modified = True

    if not modified:
        print(f"Warning: Could not find match '{match_str}' in {file_path}")
    else:
        print(f"Successfully modified {file_path}")


def repeat_sweep(sweep_name: str, n_repeats: int, autolaunch: bool = True):
    """Repeat a sweep by duplicating case directories with new case numbers.

    Creates new directories for each repeat, copies the existing case
    directories into them, and modifies ``script_uniax.py`` files to enable
    random orientation.  Existing ``.rocky``, slurm output, and log files
    are deleted to ensure a clean slate.  If ``autolaunch`` is ``True``,
    the simulations are automatically submitted via ``sbatch``.

    .. note::
       Only sweeps originally generated via the ``rocky_prepost`` backend
       are currently supported.

    Args:
        sweep_name: Name of the existing sweep directory to repeat.
        n_repeats: Number of times to repeat the sweep.
        autolaunch: If ``True``, automatically submit SLURM jobs after
            setup. Defaults to ``True``.

    Raises:
        NotADirectoryError: If the sweep directory does not exist.
    """
    # Resolve absolute path to handle trailing slashes and relative paths correctly
    sweep_path = os.path.abspath(sweep_name)
    parent_dir = os.path.dirname(sweep_path)
    base_name = os.path.basename(sweep_path)

    if not os.path.isdir(sweep_path):
        raise NotADirectoryError(f"Sweep directory not found at {sweep_path}")

    current_repeat = 0
    while current_repeat < n_repeats:
        # Construct new path using the parent directory and clean base name
        new_dir = os.path.join(parent_dir, f"{base_name}_repeat_{current_repeat + 1}")

        # Copy files
        if os.path.exists(new_dir):
            shutil.rmtree(new_dir)
        shutil.copytree(sweep_path, new_dir)

        current_repeat += 1

        # Delete simulation files for all cases in the new sweep
        rocky_files = glob.glob(os.path.join(new_dir, "case_*", "*.rocky*"))
        slurm_files = glob.glob(os.path.join(new_dir, "case_*", "slurm-*.out"))
        log_files = glob.glob(os.path.join(new_dir, "case_*", "*.log"))

        for f in rocky_files + slurm_files + log_files:
            if os.path.isfile(f):
                os.remove(f)
            elif os.path.isdir(f):
                shutil.rmtree(f)

        py_scripts = glob.glob(os.path.join(new_dir, "case_*", "script_uniax.py"))

        if not py_scripts:
            print(f"Warning: No script_uniax.py files found in {new_dir}")

        for script in py_scripts:
            _insert_line_in_file(
                file_path=script,
                match_str="# Instantiate the shape for the particle",
                new_line="particle.EnableRandomOrientation()",
            )

        if autolaunch:
            cases = glob.glob(os.path.join(new_dir, "case_*"))
            for case in cases:
                with cd(case):
                    # Launch with sbatch
                    output = subprocess.run(
                        ["sbatch", "runRocky.sh"],
                        capture_output=True,
                        text=True,
                    )
                    if output.returncode == 0:
                        print(f"Launched simulation in {case}: {output.stdout.strip()}")
                    else:
                        print(
                            f"Error launching simulation in {case}: {output.stderr.strip()}"
                        )

    print(f"Sweep '{sweep_name}' repeated {n_repeats} times.")
