import itertools
import multiprocessing
import os
import secrets
import traceback
from datetime import datetime
from typing import Any, Callable, Iterator, Mapping, Sequence, Tuple

import gin
from absl import app, flags
from acme.utils import loggers
from tqdm import tqdm

if __name__ == "__main__":
    flags.DEFINE_multi_string(
        "gin_file", None, "List of paths to the config files.")
    flags.DEFINE_multi_string(
        "gin_param", None, "Newline separated list of Gin parameter bindings.")
    flags.DEFINE_integer(
        "num_seeds", 1, "Number of different seeds to run.")

flags.DEFINE_string(
    "results_dir", "./results", "Directory used to log results.")
flags.DEFINE_integer(
    "max_concurrent",
    1,
    "Maximum number of concurrent runs.",
)
FLAGS = flags.FLAGS


def as_gin_params(config: Mapping[str, str]) -> Sequence[str]:
    params = []
    for name, value in config.items():
        value_template = "{}"
        if isinstance(value, str):
            value_template = f"'{value_template}'"
        params.append(f"{name} = {value_template.format(value)}")
    return params


@gin.configurable
def run(run_id: int, target: Callable, log_dir: str, logger):
    target(run_id, log_dir=log_dir, logger=logger)


def start(packed_configs):
    run_id, log_dir, config, gin_configs, gin_params = packed_configs

    gin_params = gin_params or []
    gin_params += as_gin_params(config)

    gin.parse_config(gin_configs)
    gin.parse_config(gin_params)
    gin.finalize()

    log_dir = os.path.join(log_dir, f"{run_id}")
    config_logger = loggers.CSVLogger(log_dir, "config")

    logger = loggers.CSVLogger(log_dir, "results")
    try:
        run(run_id=run_id, log_dir=log_dir, logger=logger)
    except Exception as e:
        print(f"ERROR: run {run_id} failed with exception {e}")
        traceback.print_exc()
        raise e

    config["run_id"] = run_id
    config["operative_config"] = gin.operative_config_str()
    config_logger.write(config)


def generate_parameters(
        config: Mapping[str, Sequence[Any]],
        log_dir: str,
        gin_configs: Sequence[str],
        gin_params: Sequence[str]
        ) -> Iterator[Tuple[int, str, Mapping[str, str], Sequence[str], Sequence[str]]]:
    names = list(config.keys())
    all_values = list(config.values())
    for i, values in enumerate(itertools.product(*all_values)):
        yield i, log_dir, dict(zip(names, values)), gin_configs, gin_params


def launch(hyperparams, gin_configs, gin_params):
    log_dir = os.path.join(FLAGS.results_dir, f"{datetime.now()}")

    all_configs = list(generate_parameters(hyperparams, log_dir, gin_configs, gin_params))
    with multiprocessing.Pool(FLAGS.max_concurrent, maxtasksperchild=1) as pool:
        outputs = pool.imap_unordered(start, all_configs, chunksize=1)
        for _ in tqdm(outputs, total=len(all_configs), dynamic_ncols=True):
            pass
