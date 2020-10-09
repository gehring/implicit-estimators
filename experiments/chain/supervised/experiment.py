from typing import NamedTuple

import gin
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tjax.annotations import PyTree

from experiments.chain import env

from imprl import modules
from imprl.mdp import value

gin.external_configurable(value.identity_offset)
gin.external_configurable(value.mean_offset)
gin.external_configurable(value.logsumexp_reduce)
gin.external_configurable(value.max_reduce)
gin.external_configurable(value.ValueIteration)

gin.external_configurable(modules.ExplicitMDP)
gin.external_configurable(modules.ExplicitWeights)
gin.external_configurable(modules.LinearModule)
gin.external_configurable(modules.MDPSolveWeights)
gin.external_configurable(modules.ExplicitMatrixWeights)

gin.external_configurable(optax.sgd)
gin.external_configurable(optax.adam)


class TrainState(NamedTuple):
    params: PyTree
    opt_state: PyTree


class TrainBatch(NamedTuple):
    inputs: PyTree
    labels: PyTree


@gin.configurable
def supervised_chain_dataset(value_solver):
    chain = env.create_chain_mdp()
    optimal_values = value_solver(jnp.zeros((chain.num_states(),)), chain)
    return jnp.eye(chain.num_states()), optimal_values


@gin.configurable
def batch_generator(key, data, batch_size, replace=True):
    inputs, labels = data
    while True:
        key, batch_key = jax.random.split(key)
        idx = jax.random.choice(batch_key, labels.shape[0], (batch_size,), replace=replace)
        yield TrainBatch(inputs[idx], labels[idx])


@gin.configurable
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

    logger.write({"timestep": 0, "eval_loss": float(evaluate(train_state.params))})

    for i, batch in enumerate(batch_gen):
        train_state, batch_loss = update(train_state, batch)

        if (i + 1) % eval_period == 0:
            eval_loss = evaluate(train_state.params)
            logger.write({"timestep": i + 1, "eval_loss": float(eval_loss)})

        if i >= num_iterations:
            break

    return train_state.params


@gin.configurable
def run_loop(run_id: int, log_dir: str, logger, seed: int):
    train_key, data_key = [s.generate_state(2) for s in np.random.SeedSequence(seed).spawn(2)]
    batch_gen = batch_generator(key=data_key)
    train(key=train_key, batch_gen=batch_gen, logger=logger)
