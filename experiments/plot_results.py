import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

from experiments import utils

font_color = "black"
sns.set_theme(
    font_scale=1.4,
    rc={"text.color": font_color,
        "axes.labelcolor": font_color,
        "xtick.color": font_color,
        "ytick.color": font_color},
)

# need to rerun:
# - mountain car implicit lr: 0.125 - 4.
# - mountain car explicit lr: 0.25 - 8.

COL_NAME = "Parameterization"

ylims = {
    "chain_rollout": (0., 15.),
    "four_rooms_rollout": (0., 50.),
    "mountaincar": (0., 700.),
}

RUN_IDS = {
    ("chain_rollout", "explicit"): 202105202021,
    ("chain_rollout", "implicit"): 202105261803,
    # ("chain_rollout", "implicit"): 202105221803,
    ("four_rooms_rollout", "explicit"): 202105241245,
    ("four_rooms_rollout", "implicit"): 202105231238,
    ("mountaincar", "explicit"): 202105261546,
    ("mountaincar", "implicit"): 202105261313,
}

BEST_HPARAMS = {
    ("chain_rollout", "explicit"): [{"LEARNING_RATE": 2 ** -1}],
    ("chain_rollout", "implicit"): [{"LEARNING_RATE": 2 ** -5, "MDP_MODULE_DISCOUNT": 0.8},
                                    {"LEARNING_RATE": 2 ** -6, "MDP_MODULE_DISCOUNT": 0.9},
                                    {"LEARNING_RATE": 2 ** -6, "MDP_MODULE_DISCOUNT": 0.95},
                                    {"LEARNING_RATE": 2 ** -7, "MDP_MODULE_DISCOUNT": 0.975},
                                    {"LEARNING_RATE": 2 ** -8, "MDP_MODULE_DISCOUNT": 0.99}],
    ("four_rooms_rollout", "explicit"): [{"LEARNING_RATE": 2 ** 5}],
    ("four_rooms_rollout", "implicit"): [{"LEARNING_RATE": 2**2, "MDP_MODULE_DISCOUNT": 0.8},
                                         {"LEARNING_RATE": 2**1, "MDP_MODULE_DISCOUNT": 0.9},
                                         {"LEARNING_RATE": 1., "MDP_MODULE_DISCOUNT": 0.95},
                                         {"LEARNING_RATE": 2**-2, "MDP_MODULE_DISCOUNT": 0.975}],
    ("mountaincar", "explicit"): [{"LEARNING_RATE": 2**4}],  #
    ("mountaincar", "implicit"): [{"LEARNING_RATE": 2**2, "MDP_MODULE_DISCOUNT": 0.8},
                                  {"LEARNING_RATE": 2**1, "MDP_MODULE_DISCOUNT": 0.9},
                                  {"LEARNING_RATE": 1., "MDP_MODULE_DISCOUNT": 0.95},
                                  {"LEARNING_RATE": 2**-2, "MDP_MODULE_DISCOUNT": 0.975}],
}

TRUE_DISCOUNTS = {
    "chain_rollout": 0.9,
    "four_rooms_rollout": 0.9,
    # "mountaincar": 0.99,
}

PRETTY_DOMAIN_NAMES = {
    "chain_rollout": "Chain MDP",
    "four_rooms_rollout": "Foor Rooms",
    "mountaincar": "Mountain Car",
}

result_path = "~/results/{domain}/{method}/{id:d}"


def load_all_results(run_ids):
    results = {}
    for domain, method in run_ids:
        path = result_path.format(domain=domain, method=method, id=run_ids[(domain, method)])
        df = utils.load_results(path)
        results[(domain, method)] = df

    return results


def plot_start_residual(results,
                        use_bias=False,
                        batch_size=25,
                        label_name="Component",
                        palette=None):
    col_name = "Domain"
    plot_dfs = []
    for domain in ylims:
        # both explicit and implicit are initialized the same so we just plot explicit
        df = results[(domain, "explicit")]

        # only consider the first residual
        mask = df.timestep == df.timestep.min()

        # select only a single set of hyperparameters
        mask = mask & (df.USE_BIAS == use_bias)
        mask = mask & (df.LEARNING_RATE == df.LEARNING_RATE.min())
        mask = mask & (df.BATCH_SIZE == batch_size)

        # the differences of initial residual between different module discount are minor
        mask &= df.MDP_MODULE_DISCOUNT == df.MDP_MODULE_DISCOUNT.min()

        df = df[mask].copy()
        df[col_name] = PRETTY_DOMAIN_NAMES[domain]
        df = df.melt(id_vars=["REWARD_OFFSET", col_name],
                     value_vars=["res_ones_norm", "res_ortho_norm"],
                     var_name=label_name)
        plot_dfs.append(df)

    plot_dfs = pd.concat(plot_dfs, ignore_index=True)
    plot_dfs.replace(
        {
            "res_ones_norm": r"$r^\parallel(\hat\theta)$",
            "res_ortho_norm": r"$||\mathbf{r}^\perp(\hat\theta)||$",
        },
        inplace=True,
    )

    facet_kws = {"margin_titles": True, "sharey": False}
    fig = sns.relplot(
        data=plot_dfs,
        x="REWARD_OFFSET",
        y="value",
        hue=label_name,
        col=col_name,
        palette=palette,
        kind="line",
        ci="sd",
        legend="full",
        facet_kws=facet_kws,
    )
    fig.set_titles(col_template=r"{col_name}")
    fig.set_axis_labels("Reward Offset", r" ")
    fig.savefig(f"initial_norm.pdf")


def plot_all_end_residual(results, use_bias=False, batch_size=25, palette=None):
    row_name = "Domain"
    style_name = "Best"
    label_name = "Learning Rate"

    for method in ["explicit", "implicit"]:
        for domain in ylims:
            df = results[(domain, method)]

            # only consider the last residual
            mask = df.timestep == df.timestep.max()
            mask = mask & (df.USE_BIAS == use_bias)
            mask = mask & (df.BATCH_SIZE == batch_size)

            df = df[mask].copy()
            df[row_name] = domain
            df[style_name] = utils.compute_best_mask(df, BEST_HPARAMS[(domain, method)])
            df.rename(columns={"LEARNING_RATE": label_name}, inplace=True)

            facet_kws = {
                "margin_titles": True,
                "sharey": False,
                "ylim": ylims[domain],
            }
            fig = sns.relplot(
                data=df,
                x="REWARD_OFFSET",
                y="residual_norm",
                hue=label_name,
                style=style_name,
                palette=palette,
                col="MDP_MODULE_DISCOUNT",
                kind="line",
                ci="sd",
                legend="full",
                facet_kws=facet_kws,
            )
            fig.set_titles(col_template=r"$\eta = {col_name}$")
            fig.set_axis_labels("Reward Offset", r"$||\mathbf{r}||_2$")
            fig.savefig(f"{method}_{domain}_norm.pdf")


def plot_all_eta_compare_residual(results, use_bias=False, batch_size=25, palette=None):
    row_name = "Domain"
    hue_name = "Parameterization"

    for domain in ylims:
        plot_dfs = []
        for method in ["explicit", "implicit"]:
            df = results[(domain, method)]

            # only consider the last residual
            mask = df.timestep == df.timestep.max()
            mask = mask & (df.USE_BIAS == use_bias)
            mask = mask & (df.BATCH_SIZE == batch_size)

            df = df[mask]
            # filter again to keep only data from the best hparams
            df = df[utils.compute_best_mask(df, BEST_HPARAMS[(domain, method)])]

            df = df.copy()
            df[row_name] = domain
            df[hue_name] = method
            plot_dfs.append(df)

        plot_dfs = pd.concat(plot_dfs, ignore_index=True)

        facet_kws = {
            "margin_titles": True,
            "sharey": False,
        }
        fig = sns.relplot(
            data=plot_dfs,
            x="REWARD_OFFSET",
            y="residual_norm",
            hue=hue_name,
            palette=palette,
            col="MDP_MODULE_DISCOUNT",
            kind="line",
            ci="sd",
            legend="full",
            facet_kws=facet_kws,
        )
        fig.set_titles(col_template=r"$\eta = {col_name}$")
        fig.set_axis_labels("Reward Offset", r"$||\mathbf{r}||_2$")
        fig.savefig(f"compare_{domain}_norm.pdf")


def plot_compare_end_residual(results, use_bias=False, batch_size=25, eta=0.95, palette=None):
    col_name = "Domain"
    hue_name = "Parameterization"

    plot_dfs = []
    for domain in ylims:
        for method in ["explicit", "implicit"]:
            df = results[(domain, method)]

            # only consider the last residual
            mask = df.timestep == df.timestep.max()
            mask = mask & (df.USE_BIAS == use_bias)
            mask = mask & (df.BATCH_SIZE == batch_size)
            mask = mask & (df.MDP_MODULE_DISCOUNT == eta)

            df = df[mask]
            # filter again to keep only data from the best hparams
            df = df[utils.compute_best_mask(df, BEST_HPARAMS[(domain, method)])]

            df = df.copy()
            df[col_name] = PRETTY_DOMAIN_NAMES[domain]
            df[hue_name] = method
            plot_dfs.append(df)

    plot_dfs = pd.concat(plot_dfs, ignore_index=True)

    facet_kws = {
        "margin_titles": True,
        "sharey": False,
    }
    fig = sns.relplot(
        data=plot_dfs,
        x="REWARD_OFFSET",
        y="residual_norm",
        hue=hue_name,
        palette=palette,
        col=col_name,
        kind="line",
        ci="sd",
        legend="full",
        facet_kws=facet_kws,
    )
    fig.set_titles(col_template=r"{col_name}")
    fig.set_axis_labels("Reward Offset", r"$||\mathbf{r}||_2$")
    fig.savefig(f"compare_norm.pdf")


def _annotate_true_discount(data, **kwargs):
    del kwargs

    domains = data.Domain.unique()
    assert len(domains) == 1
    inv_pretty_map = dict([(v, k) for k, v in PRETTY_DOMAIN_NAMES.items()])
    discount = TRUE_DISCOUNTS.get(inv_pretty_map[domains[0]])

    if discount is not None:
        ax = plt.gca()
        ax.axvline(discount, color="k", alpha=0.6, linestyle="--", zorder=1.)


def plot_eta_end_residual(results, use_bias=False, batch_size=25, palette=None):
    col_name = "Domain"
    label_name = "Reward Offset"
    method = "implicit"
    eta_vals = [0.8, 0.9, 0.95, 0.975]

    plot_dfs = []
    for domain in ylims:
        df = results[(domain, method)]

        # only consider the last residual
        mask = df.timestep == df.timestep.max()
        mask = mask & (df.USE_BIAS == use_bias)
        mask = mask & (df.BATCH_SIZE == batch_size)

        df = df[mask]
        # filter again to keep only data from the best hparams
        df = df[utils.compute_best_mask(df, BEST_HPARAMS[(domain, method)])]

        df = df.copy()
        df[col_name] = PRETTY_DOMAIN_NAMES[domain]
        plot_dfs.append(df)

    plot_dfs = pd.concat(plot_dfs, ignore_index=True)
    plot_dfs.rename(columns={"REWARD_OFFSET": label_name}, inplace=True)

    facet_kws = {
        "margin_titles": True,
        "sharey": False,
    }
    g = sns.relplot(
        data=plot_dfs,
        x="MDP_MODULE_DISCOUNT",
        y="residual_norm",
        col=col_name,
        hue=label_name,
        palette=palette,
        markers=["o"]*11,
        style=label_name,
        dashes=False,
        kind="line",
        ci=None,
        legend="full",
        facet_kws=facet_kws,
    )
    g.set(xscale="logit")
    g.set(xticks=eta_vals)
    g.set(xticklabels=[str(x) for x in eta_vals])
    g.map_dataframe(_annotate_true_discount)

    g.set_titles(col_template=r"{col_name}")
    g.set_axis_labels(r"$\eta$", r"$||\mathbf{r}||_2$")

    g.savefig(f"eta_norm.pdf")


print("Loading data...")
results = load_all_results(RUN_IDS)

print("Plotting start residual figures...")
plot_start_residual(results, palette=sns.color_palette("colorblind", 4)[2:])

print("Plotting all end residual figures...")
plot_all_end_residual(results)

print("Plotting eta end residual figures...")
plot_eta_end_residual(results, palette=sns.color_palette("ch:s=-.2,r=.6,l=.75", as_cmap=True))

print("Plotting end residual comparison figures...")
plot_compare_end_residual(results, palette=sns.color_palette("colorblind", 4)[:2])

print("Plotting end residual comparison figures for all etas...")
plot_all_eta_compare_residual(results)
