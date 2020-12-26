import ast
from concurrent import futures
import glob
import itertools
import os

import pandas as pd


def load_run_v1(run_dir, config_dir):
    log_path = os.path.join(run_dir, "logs.csv")
    config_path = os.path.join(config_dir, os.path.basename(run_dir) + ".gin")

    try:
        df = pd.read_csv(log_path)

        with open(config_path, "r") as f:
            lines = f.readlines()

    except FileNotFoundError:
        return None

    for line in lines:
        name, value = line.split("=")
        df[name.strip()] = ast.literal_eval(value.strip())

    return df


def load_run_v2(run_dir, config_dir=None):
    del config_dir
    run_dir = os.path.normpath(run_dir)
    run_id = int(os.path.basename(run_dir))
    log_path = os.path.join(run_dir, "logs.csv")

    try:
        df = pd.read_csv(log_path)
    except FileNotFoundError:
        return None

    df["run_id"] = run_id

    return df


load_run = load_run_v2


def load_results(result_dir, config_dir=None):
    result_dir = os.path.expanduser(result_dir)
    with futures.ThreadPoolExecutor() as executor:
        dfs = executor.map(
            load_run, glob.glob(os.path.join(result_dir, "*/")), itertools.repeat(config_dir))

    df = pd.concat([df for df in dfs if df is not None], ignore_index=True)

    if config_dir is None:
        joblogs = pd.read_csv(os.path.join(result_dir, "results.csv"))
        joblogs = joblogs.rename(columns={"Seq": "run_id"})
        joblogs = joblogs.drop(
            columns=["Host", "Starttime", "JobRuntime", "Send", "Receive", "Exitval", "Signal",
                     "Command", "Stdout", "Stderr"],
        )
        df = df.merge(joblogs)

    return df
