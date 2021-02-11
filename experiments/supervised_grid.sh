#!/bin/bash

# for i in {1..20}; do echo $(od -A n -t u -N 4 /dev/urandom | tr -d ' \n'); done
SEEDS="733348321 4017802264 384799037 3602412539 107577988
       2182315642 4173556719 2390141961 1953635580 3390822559
       991818103 1395720285 2034693659 240309497 2289609983
       3543305936 90715284 1111384852 1501843923 4047805392"

agent_config_path=$1
agent_name=$(basename $agent_config_path .gin)

env_config_path=$2
env_name=$(basename $env_config_path .gin)

results_dir=${3:-results}/$env_name/$agent_name/$(date +%Y%m%d%H%M)

mkdir -p $results_dir

parallel --nice 10 --memfree 3G --linebuffer --shuf --joblog $results_dir/joblog --header : --results $results_dir/results.csv \
    "taskset -c "'$(({%} - 1 ))'" python supervised.py \
      --gin_file $agent_config_path \
      --gin_file $env_config_path \
      --gin_param SEED={SEED} \
      --gin_param LEARNING_RATE={LEARNING_RATE} \
      --gin_param BATCH_SIZE={BATCH_SIZE} \
      --gin_param MDP_MODULE_DISCOUNT={MDP_MODULE_DISCOUNT} \
      --results_dir $results_dir/{#}/" \
    ::: SEED $SEEDS \
    ::: LEARNING_RATE $(for i in {0..4}; do bc -l <<< "2^(-$i + 1)"; done) \
    ::: BATCH_SIZE 1 5 25 \
    ::: MDP_MODULE_DISCOUNT 0.8 0.9 0.95
