import seaborn as sns
import matplotlib.pyplot as plt

from experiments import utils

sns.set_theme()

results_path = {
    "implicit": {
        "chain": "~/results/chain/implicit/202101111222",
        "four_rooms": "~/results/four_rooms/implicit/202101121451",
        "mountain_car": "~/results/mountaincar/implicit/202101121507",
    },
    "explicit": {
        "chain": "~/results/chain/explicit/202101051403",
        "four_rooms": "~/results/four_rooms/explicit/202101051401",
        "mountain_car": "~/results/mountaincar/explicit/202101121604",
    }
}

best_hparams = {
    "explicit": {
        "chain": [{"BATCH_SIZE": 1, "LEARNING_RATE": 1.},
                  {"BATCH_SIZE": 5, "LEARNING_RATE": 2.},
                  {"BATCH_SIZE": 25, "LEARNING_RATE": 2.}],
        "four_rooms": [{"BATCH_SIZE": 1, "LEARNING_RATE": 1.},
                       {"BATCH_SIZE": 5, "LEARNING_RATE": 2.},
                       {"BATCH_SIZE": 25, "LEARNING_RATE": 2.}],
        "mountain_car": [{"LEARNING_RATE": 2.}],
    },
    "implicit": {
        "chain": [{"BATCH_SIZE": 1, "MDP_MODULE_DISCOUNT": 0.8, "LEARNING_RATE": 0.25},
                  {"BATCH_SIZE": 1, "MDP_MODULE_DISCOUNT": 0.9, "LEARNING_RATE": 0.125},
                  {"BATCH_SIZE": 5, "MDP_MODULE_DISCOUNT": 0.8, "LEARNING_RATE": 0.5},
                  {"BATCH_SIZE": 5, "MDP_MODULE_DISCOUNT": 0.9, "LEARNING_RATE": 0.25},
                  {"BATCH_SIZE": 25, "MDP_MODULE_DISCOUNT": 0.8, "LEARNING_RATE": 0.5},
                  {"BATCH_SIZE": 25, "MDP_MODULE_DISCOUNT": 0.9, "LEARNING_RATE": 0.25},
                  {"MDP_MODULE_DISCOUNT": 0.95, "LEARNING_RATE": 0.0625}],
        "four_rooms": [{"BATCH_SIZE": 1, "MDP_MODULE_DISCOUNT": 0.8, "LEARNING_RATE": 0.25},
                       {"BATCH_SIZE": 1, "MDP_MODULE_DISCOUNT": 0.9, "LEARNING_RATE": 0.25},
                       {"BATCH_SIZE": 1, "MDP_MODULE_DISCOUNT": 0.95, "LEARNING_RATE": 0.125},
                       {"BATCH_SIZE": 5, "LEARNING_RATE": 0.5},
                       {"BATCH_SIZE": 25, "LEARNING_RATE": 0.5}],
        "mountain_car": [{"BATCH_SIZE": 1, "MDP_MODULE_DISCOUNT": 0.8, "LEARNING_RATE": 0.5},
                         {"BATCH_SIZE": 1, "MDP_MODULE_DISCOUNT": 0.9, "LEARNING_RATE": 0.5},
                         {"BATCH_SIZE": 1, "MDP_MODULE_DISCOUNT": 0.95, "LEARNING_RATE": 0.25},
                         {"BATCH_SIZE": 5, "LEARNING_RATE": 0.5},
                         {"BATCH_SIZE": 25, "LEARNING_RATE": 0.5}],
    }
}


def plot_domain_results(domain, ylim):
    # load and plot implicit chain results
    implicit_df = utils.load_results(results_path["implicit"][domain])
    fig = utils.plot_results(
        data=implicit_df,
        ylim=ylim,
        best_hparams=best_hparams["implicit"]["chain"],
    )
    fig.savefig(f"implicit_{domain}_norm.pdf")

    # load and plot explicit chain results
    explicit_df = utils.load_results(results_path["explicit"][domain])
    fig = utils.plot_results(
        data=explicit_df,
        ylim=ylim,
        best_hparams=best_hparams["explicit"]["chain"],
    )
    fig.savefig(f"explicit_{domain}_norm.pdf")

    # filter by best hyper-parameters
    best_implicit = utils.compute_best_mask(implicit_df, best_hparams["implicit"][domain])
    best_explicit = utils.compute_best_mask(explicit_df, best_hparams["explicit"][domain])

    # plot comparisons for the chain results
    fig = utils.plot_comparison(
        implicit_df[best_implicit],
        explicit_df[best_explicit],
        y="residual_norm",
        ylim=ylim,
    )
    fig.savefig(f"compare_{domain}_norm.pdf")

    fig = utils.plot_comparison(
        implicit_df[best_implicit],
        explicit_df[best_explicit],
        y="res_ones_norm",
        ylim=ylim,
    )
    fig.savefig(f"compare_{domain}_ones_norm.pdf")


# load and plot chain results
plot_domain_results("chain", ylim=(0., 40.))

# load and plot four rooms results
plot_domain_results("four_rooms", ylim=(0., 5.))

# load and plot mountain car results
plot_domain_results("mountain_car", ylim=(0., 1900.))
