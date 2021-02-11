from concurrent import futures
import glob
import itertools
import os

import pandas as pd

import seaborn as sns


def load_run(run_dir, config_dir=None):
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
            errors="ignore",
        )
        df = df.merge(joblogs)

    return df


def compute_best_mask(data, best_hparams):
    """Compute mask to select data corresponding to best hyper-parameters.

    `best_hparams` is assumed to be a list of combinations of best hyper-parameters. The returned
    mask is the `OR` of the mask corresponding to each combination. A mask for a given combination
    is corresponds to the `AND` when selecting for `NAME == VALUE` for each name/value pair.
    """
    best_mask = False
    for term in best_hparams:
        term_mask = True
        for name, value in term.items():
            term_mask &= data[name] == value
        best_mask |= term_mask
    return best_mask


def plot_results(data, ylim=None, best_hparams=None, y="residual_norm", hue="LEARNING_RATE"):
    facet_kws = {"margin_titles": True}
    if ylim is not None:
        facet_kws["ylim"] = ylim

    style = None
    if best_hparams:
        style = "BEST"
        data = data.copy()
        data[style] = compute_best_mask(data, best_hparams)

    return sns.relplot(
        data=data,
        x="timestep",
        y=y,
        hue=hue,
        style=style,
        row="MDP_MODULE_DISCOUNT",
        col="BATCH_SIZE",
        kind="line",
        ci="sd",
        legend="full",
        facet_kws=facet_kws,
    )


def plot_comparison(implicit_data, explicit_data, y="residual_norm", ylim=None):
    label_name = "Parameterization"

    implicit_data = implicit_data.copy()
    implicit_data[label_name] = "implicit"

    explicit_data = explicit_data.copy()
    explicit_data[label_name] = "explicit"

    data = pd.concat((implicit_data, explicit_data), ignore_index=True)
    return plot_results(data, ylim=ylim, y=y, hue=label_name)
