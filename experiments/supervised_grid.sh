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

lr_offset=$3

results_dir=${4:-"${HOME}/results"}/$env_name/$agent_name/$(date +%Y%m%d%H%M)

export JAX_ENABLE_X64=True

mkdir -p $results_dir

parallel --nice 10 --memfree 3G --linebuffer --shuf --retries 3 --joblog $results_dir/joblog --header : --results $results_dir/results.csv \
    "taskset -c "'$(({%} - 1 ))'" ./run_supervised.sh \
      --gin_file $agent_config_path \
      --gin_file $env_config_path \
      --gin_param SEED={SEED} \
      --gin_param LEARNING_RATE={LEARNING_RATE} \
      --gin_param BATCH_SIZE={BATCH_SIZE} \
      --gin_param MDP_MODULE_DISCOUNT={MDP_MODULE_DISCOUNT} \
      --gin_param REWARD_OFFSET={REWARD_OFFSET} \
      --gin_param USE_BIAS={USE_BIAS} \
      --results_dir $results_dir/{#}/" \
    ::: SEED $SEEDS \
    ::: LEARNING_RATE $(for i in {0..5}; do bc -l <<< "2^(-$i + $lr_offset)"; done) \
    ::: BATCH_SIZE 25 \
    ::: MDP_MODULE_DISCOUNT 0.8 0.9 0.95 0.975 \
    ::: REWARD_OFFSET 0. 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 \
    ::: USE_BIAS False
