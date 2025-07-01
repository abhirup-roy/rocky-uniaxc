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

    db_path = os.path.join(
        PWD, sweep_name, 'results.db'
    )

    if not os.path.isfile(db_path):
        raise FileNotFoundError(f'Results db not found at {db_path}')

    read_table_sql = 'SELECT * FROM results;'
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(read_table_sql, conn, index_col='id')

    return df


def dump_db(
        sweep_name: str,
        filetype: str = 'csv',
        outputs_dir: str = 'pyoutputs'):
    
    outputs_dir_path = os.path.join(
        PWD, sweep_name, outputs_dir
    )
    if not os.path.isdir(outputs_dir):
        os.makedirs(outputs_dir_path, exist_ok=True)

    df = load_data(sweep_name=sweep_name)

    if filetype == 'csv':
        dump_path = os.path.join(
            outputs_dir_path, 'results.csv'
        )
        df.to_csv(dump_path)
    elif filetype == 'parquet':
        dump_path = os.path.join(
            outputs_dir_path, 'results.parquet.gzip'
        )
        df.to_parquet(dump_path, compression='gzip')
    elif filetype == 'feather':
        dump_path = os.path.join(
            outputs_dir_path, 'results.feather'
        )
        df.to_feather(dump_path)


def find_faulty_runs(sweep_name: str, dump: bool = False):
    sweep_dir = os.path.join(PWD, sweep_name)
    # Get all subdirectories that start with 'case_'
    case_dirs = [
        entry.name for entry in os.scandir(sweep_dir)
        if entry.is_dir() and entry.name.startswith('case_')
    ]

    faulty_cases = {}
    for case_dir in case_dirs:
        case_path = os.path.join(sweep_dir, case_dir)
        results_file = os.path.join(case_path, 'results.csv')

        if not os.path.isfile(results_file):
            continue
        else:
            ser = pd.read_csv(results_file).squeeze()
            if ser['hausner_ratio'] < 1 or ser['hausner_ratio'] > 3:
                faulty_cases[case_dir] = ser['hausner_ratio']
                
        
        output_file = glob.glob(
            os.path.join(case_path, 'slurm-*.out')
        )
        if output_file:
            with open(output_file[0], 'r') as f:
                content = f.read()
                if "RuntimeWarning: Particles were lost during the simulation" in content:

                    particles_init = int(re.search(
                        r"Initial particle count: (\d+)", content).group(1))
                    particles_final = int(re.search(
                        r"Final particle count: (\d+)", content).group(1))
                    print(f"Warning: Particles lost in {case_dir}: {particles_init} -> {particles_final} ({particles_final - particles_init})")

    print(f"Faulty cases in sweep '{sweep_name}':")

    if faulty_cases:
        for case, hr_value in faulty_cases.items():
            print(f"  - {case}:   HR = {hr_value:.3f}")
    else:
        print("  No faulty cases found.")
    if dump:
        with open(os.path.join(sweep_dir, 'faulty_cases.txt'), 'w') as f:
            for case in faulty_cases.items():
                f.write(f"{case}\n")
            print(
                f"  Faulty cases dumped to {os.path.join(sweep_dir, 'faulty_cases.txt')}")


if __name__ == "__main__":
    
    sweep_name = "test"

    dump_db(sweep_name=sweep_name)
    find_faulty_runs(sweep_name=sweep_name)
