# Experiments

Experiments we ran on an AMD Ryzen 5950X CPU with fast memory. Each run was confined to run on a single vCPU using linux cgroups.

## Running the code
**The experiment code assumes it is being called from the `experiments` directory.**

All experiments are run with `python supervised.py` using different configuration using [`gin`](https://github.com/google/gin-config) config files. Since the config file may contain unused values and some config value might be hidden as defaults, the exact config values used are part of the output of the experiments, e.g., `results/implicit/202102021347/1/config.gin`. A specific run can be run again using:
```commandline
python supervised.py --gin_file PATH_TO_GIN_CONFIG_FILE
```

We provide a bash script which can call `supervised.py` with a gridsearch like combination of parameters. This assumes you are running with a recent version of linux and requires [GNU parallel](https://www.gnu.org/software/parallel/) to run. The learning rates are hardcoded in this script and don't necessarily reflect the learning rates tried. See the full plots of each domain to see the values of the learning rates tried. The first argument is the config for the method and the second is the config for the environment.

Example for running the explicit parameterization on mountain car:
```commandline
./supervised_grid.sh configs/supervised/explicit.gin configs/supervised/envs/mountaincar.gin 
```

## Results

The data used for our figures are under the `results` directory. We've removed any columns or files that contain identifying information. The `results.csv` files contain additional information about our runs, such as the runtimes. Our exact figures can be plotted with the following command:
```commandline
python plot_results.py
```
