#!/bin/bash

config_path=$1
job_config_dir=$2
results_dir=${3:-"~/results"}/$(basename $config_path .gin)/$(date +%Y%m%d%H%M)

export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export JAX_ENABLE_X64=True

parallel --shuf "python supervised.py \
    --gin_file $config_path \
    --gin_file $2/{1}.gin \
    --results_dir $results_dir/{1}" \
    ::: $(basename -s .gin $job_config_dir/*.gin)
