import os
from typing import NamedTuple, Optional

from absl import app
from absl import flags

import gin

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import optax

from tjax.annotations import PyTree

from imprl import modules
from imprl.mdp import value

from experiments import rollout


if __name__ == "__main__":
    flags.DEFINE_multi_string(
        "gin_file", None, "List of paths to the config files.")
    flags.DEFINE_multi_string(
        "gin_param", None, "Newline separated list of Gin parameter bindings.")
    flags.DEFINE_string(
        "results_dir", "./results", "Directory used to log results.")

FLAGS = flags.FLAGS

gin.external_configurable(value.identity_offset)
gin.external_configurable(value.mean_offset)
gin.external_configurable(value.fixed_offset)
gin.external_configurable(value.logsumexp_reduce)
gin.external_configurable(value.max_reduce)
gin.external_configurable(value.ValueIteration)

gin.external_configurable(modules.ExplicitMDP)
gin.external_configurable(modules.ExplicitWeights)
gin.external_configurable(modules.LinearModule)
gin.external_configurable(modules.MDPSolveWeights)
gin.external_configurable(modules.ExplicitMatrixWeights)
gin.external_configurable(modules.normalized_scales)
gin.external_configurable(modules.RBFEncoder)
gin.external_configurable(modules.OneHot)
gin.external_configurable(modules.uniform_centers)

gin.external_configurable(optax.sgd)
gin.external_configurable(optax.adam)


class TrainState(NamedTuple):
    params: PyTree
    opt_state: PyTree


class TrainBatch(NamedTuple):
    inputs: PyTree
    labels: PyTree


@gin.configurable
def supervised_mdp_dataset(mdp, value_solver):
    optimal_values = value_solver(jnp.zeros((mdp.num_states(),)), mdp)
    return np.eye(mdp.num_states()), np.array(optimal_values)


@gin.configurable
def supervised_mrc_dataset(key, mdp, policy):
    del key

    vprobs = jax.vmap(policy.probs)
    a_probs = vprobs(jnp.arange((mdp.num_states(),)))

    rewards = np.einsum("xa,xa->x", mdp.rewards, a_probs)
    transitions = np.einsum("xay,xa->xy", mdp.transitions[:, :, :], a_probs)
    values = np.linalg.solve(np.eye(mdp.num_states()) - mdp.discounts * transitions, rewards)

    return np.arange(mdp.num_states()), np.array(values)


@gin.configurable
def batch_generator(key, data, batch_size, replace=True):
    inputs, labels = data
    while True:
        key, batch_key = jax.random.split(key)
        idx = jax.random.choice(batch_key, labels.shape[0], (batch_size,), replace=replace)
        yield TrainBatch(np.take(inputs, idx, axis=0), np.take(labels, idx, axis=0))


@gin.configurable
def rollout_dataset(key, *,
                    env_cls,
                    policy,
                    max_traj_length: int,
                    discount: float,
                    num_traj: Optional[int] = None,
                    num_steps: Optional[int] = None,
                    use_partial_traj: bool = True):
    if num_traj == num_steps:
        raise TypeError((
            "Either `num_traj` or `num_steps` is required. Providing a value for both is not "
            "supported."
        ))

    env_seed, policy_seed = np.random.SeedSequence(key).spawn(2)
    policy_key = policy_seed.generate_state(2)

    env = env_cls(seed=env_seed.generate_state(1)[0])
    data = []

    traj_count = 0
    step_count = 0
    while ((num_traj is None or traj_count < num_traj)
           and (num_steps is None or step_count < num_steps)):

        traj_len_limit = max_traj_length
        if num_steps is not None:
            traj_len_limit = min((traj_len_limit, num_steps - step_count))

        traj_key, policy_key = jax.random.split(policy_key)
        traj, _ = rollout.generate_trajectory(traj_key, env, policy, max_steps=traj_len_limit)

        observations, returns = rollout.per_observation_discounted_returns(traj, discount)
        if not use_partial_traj:
            observations = observations[:1]
            returns = returns[:1]
        data.append((observations, returns))

        traj_count += 1
        step_count += data[-1][0].shape[0]

    observations, returns = zip(*data)
    return np.concatenate(observations), np.concatenate(returns)


@gin.configurable
def dataset_factory(key, train_dataset_cls, test_dataset_cls):
    train_key, test_key = [s.generate_state(2) for s in np.random.SeedSequence(key).spawn(2)]
    train_data = train_dataset_cls(key=train_key)
    test_data = test_dataset_cls(key=test_key)
    return train_data, test_data


@gin.configurable
def train(key, model, optimizer, batch_gen, test_data, num_iterations, eval_period=10):

    def loss(params, inputs, labels):
        predictions = model.apply(params, inputs)
        return 0.5 * jnp.mean((predictions - labels)**2)

    @jax.jit
    def evaluate(params):
        test_inputs, test_labels = test_data
        residuals = model.apply(params, test_inputs) - test_labels

        # Project on the [1 1 ... 1]^T vector and compute norm of both the projected vector and
        # orthogonal remainder.
        res_ones = jnp.ones_like(residuals) * jnp.mean(residuals)
        res_ortho = residuals - res_ones

        norm_ones = jnp.linalg.norm(res_ones)
        norm_ortho = jnp.linalg.norm(res_ortho)

        return {"residual_norm": jnp.linalg.norm(residuals),
                "res_ones_norm": norm_ones,
                "res_ortho_norm": norm_ortho,
                "mse": jnp.mean(residuals**2)}

    @jax.jit
    def update(train_state, data):
        params, opt_state = train_state
        batch_loss, param_grad = jax.value_and_grad(loss)(params, *data)

        updates, opt_state = optimizer.update(param_grad, opt_state)
        params = optax.apply_updates(params, updates)

        return TrainState(params, opt_state), batch_loss

    params = model.init(key, next(batch_gen).inputs)
    opt_state = optimizer.init(params)
    train_state = TrainState(params, opt_state)

    logs = [{"timestep": 0}]
    logs[-1].update(jax.tree_map(float, evaluate(train_state.params)))

    for i, batch in enumerate(batch_gen):
        train_state, batch_loss = update(train_state, batch)

        if (i + 1) % eval_period == 0:
            logs.append({"timestep": i + 1})
            logs[-1].update(jax.tree_map(float, evaluate(train_state.params)))

        if i >= num_iterations:
            break

    return train_state.params, logs


@gin.configurable
def launch(seed, data_factory):
    resuts_dir = os.path.expanduser(FLAGS.results_dir)
    os.makedirs(resuts_dir, exist_ok=True)

    train_key, batch_key, data_key = [
        s.generate_state(2) for s in np.random.SeedSequence(seed).spawn(3)
    ]
    train_data, test_data = data_factory(key=data_key)
    batch_gen = batch_generator(key=batch_key, data=train_data)

    _, results = train(key=train_key, batch_gen=batch_gen, test_data=test_data)

    logs_path = os.path.join(resuts_dir, "logs.csv")
    pd.DataFrame.from_records(results).to_csv(logs_path, index=False)

    config_path = os.path.join(resuts_dir, "config.gin")
    with open(config_path, "w") as f:
        f.write(gin.operative_config_str())


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    print(f"Using result dir: {FLAGS.results_dir}")
    # read config file(s).
    gin_configs = []
    for config_file in FLAGS.gin_file:
        with open(config_file, "r") as file:
            gin_configs.append(file.read())

    gin_params = FLAGS.gin_param or []
    gin.parse_config(gin_configs)
    gin.parse_config(gin_params)

    launch()


if __name__ == "__main__":
    app.run(main)
