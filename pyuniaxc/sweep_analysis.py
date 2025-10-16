#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import glob
import os
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd

PWD = os.getcwd()


def load_data(sweep_name: str) -> pd.DataFrame:
    db_path = os.path.join(PWD, sweep_name, "results.db")

    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"Results db not found at {db_path}")

    read_table_sql = "SELECT * FROM results;"
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(read_table_sql, conn, index_col="id")

    return df


def _write_df_file(df: pd.DataFrame, name: str, filetype: str, s_dir: str):
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
    """
    Dump the results DataFrame to a specified file format in the outputs directory.

    Parameters
    ----------
    sweep_name : str
        Name of the sweep directory containing the results database.
    filetype : str, optional
        The file format to dump the results. Options are 'csv', 'parquet', 'feather'.
    outputs_dir : str, optional
        The directory within the sweep directory to save the output files.
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
    """
    Fallback alternative to dump all results from individual case directories into a single file.
    Uses results.csv files in each case directory and combines them into one DataFrame.

    Parameters
    ----------
    sweep_name : str
        Name of the sweep directory containing case subdirectories.
    filetype : str, optional
        The file format to dump the results. Options are 'csv', 'parquet', 'feather', 'excel'.
    minimal : bool, optional
        If True, only include columns with more than one unique value.
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
