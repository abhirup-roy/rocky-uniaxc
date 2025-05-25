#!/usr/bin/env python3

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


def dump_db(sweep_name: str, filetype: str='csv', outputs_dir: str='pyoutputs'):
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


if __name__ == "__main__":
    dump_db("pp_sweep")



    
    
