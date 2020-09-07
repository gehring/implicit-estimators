from typing import NamedTuple

import gin
import jax
import jax.numpy as jnp
import optax
from tjax.annotations import PyTree

from experiments.chain import env
from imprl import mdp


class TrainState(NamedTuple):
    params: PyTree
    opt_state: PyTree


@gin.configurable
def supervised_chain_dataset(value_solver):
    chain = env.create_chain_mdp()
    optimal_values = value_solver(jnp.zeros((chain.num_states(),)), mdp)
    return jnp.eye(chain.num_states()), optimal_values


@gin.configurable
def batch_generator(key, data, batch_size, replace=True):
    inputs, labels = data
    while True:
        key, batch_key = jax.random.split(key)
        idx = jax.random.choice(batch_key, labels.shape[0], (batch_size,), replace=replace)
        yield inputs[idx], labels[idx]


def train(key, model, optimizer, batch_gen, test_data, num_iterations, logger,
          eval_period=10):

    def loss(params, inputs, labels):
        predictions = model.apply(params, inputs)
        return 0.5 * jnp.mean((predictions - labels)**2)

    @jax.jit
    def evaluate(params):
        return loss(params, *test_data)

    @jax.jit
    def update(train_state, data):
        params, opt_state = train_state
        batch_loss, param_grad = jax.value_and_grad(loss)(params, *data)

        updates, opt_state = optimizer.update(param_grad, opt_state)
        params = optax.apply_updates(params, updates)

        return TrainState(params, opt_state), batch_loss

    key, init_key = jax.random.split(key)
    params = model.init(key, next(batch_gen).inputs)
    opt_state = optimizer.init(params)
    train_state = TrainState(params, opt_state)

    logger.append({"timestep": 0,
                   "eval_loss": float(evaluate(train_state.params))})

    for i, batch in enumerate(batch_gen):
        train_state, batch_loss = update(train_state, batch)

        if (i + 1) % eval_period == 0:
            eval_loss = evaluate(train_state.params)
            logger.append({"timestep": i * eval_period + 1,
                           "eval_loss": float(eval_loss)})

        if i >= num_iterations:
            break

    with logger.log_with(index="config") as config_logger:
        config_logger.append({"gin_config": gin.operative_config_str()})

    return train_state.params


def generate_runs(seed, model, optimizer, num_iterations, num_runs, logger):
    key = jax.random.PRNGKey(seed)
    data = supervised_chain_dataset()

    for i in range(num_runs):
        key, run_key, batch_key = jax.random.split(key, 3)
        batch_gen = batch_generator(batch_key, data)
        with logger.log_with(extra={"run_id": i}) as run_logger:
            train(run_key, model, optimizer, batch_gen, data, num_iterations, run_logger)


def main(argv):
    pass
