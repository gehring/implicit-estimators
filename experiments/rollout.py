from typing import Callable, Optional, Sequence, Tuple

import numpy as np

import dm_env

import chex
import jax


def generate_trajectory(key: chex.PRNGKey,
                        env: dm_env.Environment,
                        policy: Callable,
                        max_steps: Optional[int] = None,
                        ) -> Tuple[Sequence[dm_env.TimeStep], Sequence]:
    t = 0
    timestep = env.reset()
    trajectory = [timestep]
    actions = []

    while (not timestep.last()) and (max_steps is None or t < max_steps):
        key, a_key = jax.random.split(key)
        action = policy(a_key, timestep.observation)
        timestep = env.step(action)

        t += 1
        trajectory.append(timestep)
        actions.append(action)

    return trajectory, actions


def discounted_returns(trajectory: Sequence[dm_env.TimeStep],
                       extra_discount: chex.Numeric = 1.,
                       ) -> chex.Numeric:
    _, rewards, discounts, _ = zip(*trajectory)
    returns = 0.
    for rew, discount in zip(rewards[:0:-1], discounts[:0:-1]):
        returns = rew + extra_discount * discount * returns
    return returns


def per_observation_discounted_returns(trajectory: Sequence[dm_env.TimeStep],
                                       extra_discount: chex.Numeric = 1.,
                                       ) -> Tuple[np.ndarray, np.ndarray]:
    _, rewards, discounts, observations = zip(*trajectory)
    returns = [0.]
    for rew, discount in zip(rewards[:0:-1], discounts[:0:-1]):
        returns.append(rew + extra_discount * discount * returns[-1])
    return np.array(observations[:-1]), np.array(returns[:0:-1])
