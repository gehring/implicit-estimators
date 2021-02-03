import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

from experiments import utils

sns.set_theme()

COL_NAME = "Parameterization"

ylims = {
    "chain": (0., 40.),
    "four_rooms": (0., 5.),
    "mountain_car": (0., 1900.),
}

results_path = {
    "chain": {
        "nobias": {
            "implicit": "~/results/chain/implicit/202102021347",
            "explicit": "~/results/chain/explicit/202102021420",
        },
        "bias": {
            "implicit": "~/results/chain/implicit/202102021424",
            "explicit": "~/results/chain/explicit/202102021422",
        },
    },
    "four_rooms": {
        "nobias": {
            "implicit": "~/results/four_rooms/implicit/202102021111",
            "explicit": "~/results/four_rooms/explicit/202102021114",
        },
        "bias": {
            "implicit": "~/results/four_rooms/implicit/202102021120",
            "explicit": "~/results/four_rooms/explicit/202102021124",
        },
    },
    "mountain_car": {
        "nobias": {
            "implicit": "~/results/mountaincar/implicit/202102020536",
            "explicit": "~/results/mountaincar/explicit/202102020608",
        },
        "bias": {
            "implicit": "~/results/mountaincar/implicit/202102020737",
            "explicit": "~/results/mountaincar/explicit/202102020835",
        },
    },
}

best_hparams = {
    "chain": {
        "nobias": {
            "implicit": [{"BATCH_SIZE": 1, "MDP_MODULE_DISCOUNT": 0.8, "LEARNING_RATE": 0.125},
                         {"BATCH_SIZE": 1, "MDP_MODULE_DISCOUNT": 0.9, "LEARNING_RATE": 0.0625},
                         {"BATCH_SIZE": 1, "MDP_MODULE_DISCOUNT": 0.95, "LEARNING_RATE": 0.015625},
                         {"BATCH_SIZE": 5, "MDP_MODULE_DISCOUNT": 0.8, "LEARNING_RATE": 0.25},
                         {"BATCH_SIZE": 5, "MDP_MODULE_DISCOUNT": 0.9, "LEARNING_RATE": 0.125},
                         {"BATCH_SIZE": 5, "MDP_MODULE_DISCOUNT": 0.95, "LEARNING_RATE": 0.03125},
                         {"BATCH_SIZE": 25, "MDP_MODULE_DISCOUNT": 0.8, "LEARNING_RATE": 0.5},
                         {"BATCH_SIZE": 25, "MDP_MODULE_DISCOUNT": 0.9, "LEARNING_RATE": 0.125},
                         {"BATCH_SIZE": 25, "MDP_MODULE_DISCOUNT": 0.95, "LEARNING_RATE": 0.03125}],
            "explicit": [{"BATCH_SIZE": 1, "LEARNING_RATE": 1.},
                         {"BATCH_SIZE": 5, "LEARNING_RATE": 2.},
                         {"BATCH_SIZE": 25, "LEARNING_RATE": 2.}],
        },
        "bias": {
            "implicit": [{"BATCH_SIZE": 1, "MDP_MODULE_DISCOUNT": 0.8, "LEARNING_RATE": 0.125},
                         {"BATCH_SIZE": 1, "MDP_MODULE_DISCOUNT": 0.9, "LEARNING_RATE": 0.0625},
                         {"BATCH_SIZE": 1, "MDP_MODULE_DISCOUNT": 0.95, "LEARNING_RATE": 0.015625},
                         {"BATCH_SIZE": 5, "MDP_MODULE_DISCOUNT": 0.8, "LEARNING_RATE": 0.25},
                         {"BATCH_SIZE": 5, "MDP_MODULE_DISCOUNT": 0.9, "LEARNING_RATE": 0.125},
                         {"BATCH_SIZE": 5, "MDP_MODULE_DISCOUNT": 0.95, "LEARNING_RATE": 0.03125},
                         {"BATCH_SIZE": 25, "MDP_MODULE_DISCOUNT": 0.8, "LEARNING_RATE": 0.25},
                         {"BATCH_SIZE": 25, "MDP_MODULE_DISCOUNT": 0.9, "LEARNING_RATE": 0.125},
                         {"BATCH_SIZE": 25, "MDP_MODULE_DISCOUNT": 0.95, "LEARNING_RATE": 0.03125}],
            "explicit": [{"BATCH_SIZE": 1, "LEARNING_RATE": 0.5},
                         {"BATCH_SIZE": 5, "LEARNING_RATE": 1.},
                         {"BATCH_SIZE": 25, "LEARNING_RATE": 1.}],
        },
    },
    "four_rooms": {
        "nobias": {
            "implicit": [{"BATCH_SIZE": 1, "MDP_MODULE_DISCOUNT": 0.8, "LEARNING_RATE": 0.25},
                         {"BATCH_SIZE": 1, "MDP_MODULE_DISCOUNT": 0.9, "LEARNING_RATE": 0.125},
                         {"BATCH_SIZE": 1, "MDP_MODULE_DISCOUNT": 0.95, "LEARNING_RATE": 0.125},
                         {"BATCH_SIZE": 5, "MDP_MODULE_DISCOUNT": 0.8, "LEARNING_RATE": 1.0},
                         {"BATCH_SIZE": 5, "MDP_MODULE_DISCOUNT": 0.9, "LEARNING_RATE": 0.5},
                         {"BATCH_SIZE": 5, "MDP_MODULE_DISCOUNT": 0.95, "LEARNING_RATE": 0.25},
                         {"BATCH_SIZE": 25, "MDP_MODULE_DISCOUNT": 0.8, "LEARNING_RATE": 2.},
                         {"BATCH_SIZE": 25, "MDP_MODULE_DISCOUNT": 0.9, "LEARNING_RATE": 2.},
                         {"BATCH_SIZE": 25, "MDP_MODULE_DISCOUNT": 0.95, "LEARNING_RATE": 0.5}],
            "explicit": [{"BATCH_SIZE": 1, "LEARNING_RATE": 1.},
                         {"BATCH_SIZE": 5, "LEARNING_RATE": 2.},
                         {"BATCH_SIZE": 25, "LEARNING_RATE": 2.}],
        },
        "bias": {
            "implicit": [{"BATCH_SIZE": 1, "MDP_MODULE_DISCOUNT": 0.8, "LEARNING_RATE": 0.125},
                         {"BATCH_SIZE": 1, "MDP_MODULE_DISCOUNT": 0.9, "LEARNING_RATE": 0.125},
                         {"BATCH_SIZE": 1, "MDP_MODULE_DISCOUNT": 0.95, "LEARNING_RATE": 0.125},
                         {"BATCH_SIZE": 5, "MDP_MODULE_DISCOUNT": 0.8, "LEARNING_RATE": 0.5},
                         {"BATCH_SIZE": 5, "MDP_MODULE_DISCOUNT": 0.9, "LEARNING_RATE": 0.5},
                         {"BATCH_SIZE": 5, "MDP_MODULE_DISCOUNT": 0.95, "LEARNING_RATE": 0.25},
                         {"BATCH_SIZE": 25, "MDP_MODULE_DISCOUNT": 0.8, "LEARNING_RATE": 1.},
                         {"BATCH_SIZE": 25, "MDP_MODULE_DISCOUNT": 0.9, "LEARNING_RATE": 1.},
                         {"BATCH_SIZE": 25, "MDP_MODULE_DISCOUNT": 0.95, "LEARNING_RATE": 0.25}],
            "explicit": [{"BATCH_SIZE": 1, "LEARNING_RATE": 0.5},
                         {"BATCH_SIZE": 5, "LEARNING_RATE": 1.},
                         {"BATCH_SIZE": 25, "LEARNING_RATE": 1.}],
        },
    },
    "mountain_car": {
        "nobias": {
            "implicit": [{"BATCH_SIZE": 1, "MDP_MODULE_DISCOUNT": 0.8, "LEARNING_RATE": 1.},
                         {"BATCH_SIZE": 1, "MDP_MODULE_DISCOUNT": 0.9, "LEARNING_RATE": 0.25},
                         {"BATCH_SIZE": 1, "MDP_MODULE_DISCOUNT": 0.95, "LEARNING_RATE": 0.125},
                         {"BATCH_SIZE": 5, "MDP_MODULE_DISCOUNT": 0.8, "LEARNING_RATE": 2.},
                         {"BATCH_SIZE": 5, "MDP_MODULE_DISCOUNT": 0.9, "LEARNING_RATE": 1.},
                         {"BATCH_SIZE": 5, "MDP_MODULE_DISCOUNT": 0.95, "LEARNING_RATE": 0.25},
                         {"BATCH_SIZE": 25, "MDP_MODULE_DISCOUNT": 0.8, "LEARNING_RATE": 2.},
                         {"BATCH_SIZE": 25, "MDP_MODULE_DISCOUNT": 0.9, "LEARNING_RATE": 2.},
                         {"BATCH_SIZE": 25, "MDP_MODULE_DISCOUNT": 0.95, "LEARNING_RATE": 0.5}],
            "explicit": [{"LEARNING_RATE": 2.}],
        },
        "bias": {
            "implicit": [{"BATCH_SIZE": 1, "MDP_MODULE_DISCOUNT": 0.8, "LEARNING_RATE": 0.5},
                         {"BATCH_SIZE": 1, "MDP_MODULE_DISCOUNT": 0.9, "LEARNING_RATE": 0.25},
                         {"BATCH_SIZE": 1, "MDP_MODULE_DISCOUNT": 0.95, "LEARNING_RATE": 0.125},
                         {"BATCH_SIZE": 5, "MDP_MODULE_DISCOUNT": 0.8, "LEARNING_RATE": 1.},
                         {"BATCH_SIZE": 5, "MDP_MODULE_DISCOUNT": 0.9, "LEARNING_RATE": 0.5},
                         {"BATCH_SIZE": 5, "MDP_MODULE_DISCOUNT": 0.95, "LEARNING_RATE": 0.25},
                         {"BATCH_SIZE": 25, "MDP_MODULE_DISCOUNT": 0.8, "LEARNING_RATE": 1.},
                         {"BATCH_SIZE": 25, "MDP_MODULE_DISCOUNT": 0.9, "LEARNING_RATE": 1.},
                         {"BATCH_SIZE": 25, "MDP_MODULE_DISCOUNT": 0.95, "LEARNING_RATE": 0.5}],
            "explicit": [{"LEARNING_RATE": 1.}],
        },
    },
}


def plot_implicit_and_explicit(prefix, paths, hparams, ylim):
    # load and plot implicit chain results
    implicit_df = utils.load_results(paths["implicit"])
    fig = utils.plot_results(
        data=implicit_df,
        ylim=ylim,
        best_hparams=hparams["implicit"],
    )
    fig.set_titles(row_template=r"$\eta = {row_name}$", col_template="Batch size = {col_name}")
    fig.set_axis_labels("Iteration", "r$||\mathbf{r}||_2$")
    fig._legend.texts[0].set_text("Learning rate")
    fig._legend.texts[-3].set_text("Best")
    fig.savefig(f"{prefix}_implicit_norm.pdf")

    # load and plot explicit chain results
    explicit_df = utils.load_results(paths["explicit"])
    fig = utils.plot_results(
        data=explicit_df,
        ylim=ylim,
        best_hparams=hparams["explicit"],
    )
    fig.set_titles(row_template=r"$\eta = {row_name}$", col_template="Batch size = {col_name}")
    fig.set_axis_labels("Iteration", r"$||\mathbf{r}||_2$")
    fig._legend.texts[0].set_text("Learning rate")
    fig._legend.texts[-3].set_text("Best")
    fig.savefig(f"{prefix}_explicit_norm.pdf")

    # filter by best hyper-parameters
    best_implicit = utils.compute_best_mask(implicit_df, hparams["implicit"])
    best_explicit = utils.compute_best_mask(explicit_df, hparams["explicit"])

    # plot comparisons for the chain results
    fig = utils.plot_comparison(
        implicit_df[best_implicit],
        explicit_df[best_explicit],
        y="residual_norm",
        ylim=ylim,
    )
    fig.set_titles(row_template=r"$\eta = {row_name}$", col_template="Batch size = {col_name}")
    fig.set_axis_labels("Iteration", r"$||\mathbf{r}||_2$")
    fig.savefig(f"{prefix}_compare_norm.pdf")

    fig = utils.plot_comparison(
        implicit_df[best_implicit],
        explicit_df[best_explicit],
        y="res_ones_norm",
        ylim=ylim,
    )
    fig.set_titles(row_template=r"$\eta = {row_name}$", col_template="Batch size = {col_name}")
    fig.set_axis_labels("Iteration", r"$r^\parallel ( \theta )$")
    fig.savefig(f"{prefix}_compare_ones_norm.pdf")


def plot_domain_results(domain, ylim):
    for use_bias, paths in results_path[domain].items():
        prefix = f"{domain}_{use_bias}"
        plot_implicit_and_explicit(prefix, paths, best_hparams[domain][use_bias], ylim)


def load_joint_dataframe(paths, hparams, filter_dict):
    # filter by best hyper-parameters
    implicit_df = utils.load_results(paths["implicit"])
    explicit_df = utils.load_results(paths["explicit"])

    best_implicit = utils.compute_best_mask(implicit_df, hparams["implicit"])
    best_explicit = utils.compute_best_mask(explicit_df, hparams["explicit"])

    for name, val in filter_dict.items():
        best_implicit &= implicit_df[name] == val
        best_explicit &= explicit_df[name] == val

    implicit_df = implicit_df[best_implicit].copy()
    implicit_df[COL_NAME] = "implicit"
    explicit_df = explicit_df[best_explicit].copy()
    explicit_df[COL_NAME] = "explicit"

    return pd.concat((implicit_df, explicit_df), ignore_index=True)


def _plot_joint_data(data, y):
    fig = sns.relplot(
        data=data,
        x="timestep",
        y=y,
        hue=COL_NAME,
        col="domain",
        kind="line",
        ci="sd",
        legend="full",
        facet_kws=dict(
            sharex=False,
            sharey=False,
            margin_titles=True,
        ),
    )
    for i, domain in enumerate(results_path.keys()):
        ax = fig.facet_axis(0, i)
        ax.set_ylim(ylims[domain])

    fig.set_titles(col_template="{col_name}")

    return fig


def plot_across_domains(use_bias, batch_size, eta):
    pretty_names = {"chain": "Chain", "four_rooms": "Four rooms", "mountain_car": "Mountain car"}
    filter_dict = {"BATCH_SIZE": batch_size, "MDP_MODULE_DISCOUNT": eta}

    plot_df = []
    for domain in results_path.keys():
        data = load_joint_dataframe(
            results_path[domain][use_bias], best_hparams[domain][use_bias], filter_dict)
        data["domain"] = pretty_names[domain]
        plot_df.append(data)

    plot_df = pd.concat(plot_df, ignore_index=True)

    fig = _plot_joint_data(plot_df, "res_ones_norm")
    fig.set_axis_labels("Iteration", r"$r^\parallel ( \theta )$")
    fig.savefig("ones_compare.pdf")

    fig = _plot_joint_data(plot_df, "res_ortho_norm")
    fig.set_axis_labels("Iteration", r"$||\mathbf{r}^\perp ( \theta )||_2$")
    fig.savefig("ortho_compare.pdf")

    fig = _plot_joint_data(plot_df, "residual_norm")
    fig.set_axis_labels("Iteration", r"$||\mathbf{r}||_2$")
    fig.savefig("norm_compare.pdf")


def _single_plot(data, y, ylim, ylabel):
    fig = plt.figure()
    ax = sns.lineplot(
        data=data,
        x="timestep",
        y=y,
        hue=COL_NAME,
        ci="sd",
        legend="full",
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    return fig


def plot_single_comparison(domain, use_bias, batch_size, eta):
    filter_dict = {"BATCH_SIZE": batch_size, "MDP_MODULE_DISCOUNT": eta}
    data = load_joint_dataframe(
        results_path[domain][use_bias], best_hparams[domain][use_bias], filter_dict)

    prefix = f"single_{domain}_{use_bias}_{batch_size}_{eta}"

    fig = _single_plot(data, "res_ones_norm", ylims[domain], r"$r^\parallel ( \theta )$")
    fig.savefig(f"{prefix}_ones.pdf")

    fig = _single_plot(
        data, "res_ortho_norm", ylims[domain], r"$||\mathbf{r}^\perp ( \theta )||_2$")
    fig.savefig(f"{prefix}_ortho.pdf")

    fig = _single_plot(data, "residual_norm", ylims[domain], r"$||\mathbf{r}||_2$")
    fig.savefig(f"{prefix}_norm.pdf")


# load and plot a negative result showing high variance of implicit parameterization
plot_single_comparison("chain", use_bias="nobias", batch_size=1, eta=0.95)

# load and plot cross domain comparison of residual along the ones vector
plot_across_domains(use_bias="nobias", batch_size=25, eta=0.8)

# load and plot chain results
plot_domain_results("chain", ylim=ylims["chain"])

# load and plot four rooms results
plot_domain_results("four_rooms", ylim=ylims["four_rooms"])

# load and plot mountain car results
plot_domain_results("mountain_car", ylim=ylims["mountain_car"])
