import functools
from typing import  Optional, Protocol, Sequence, Tuple

import gin

import numpy as np

import dm_env
from dm_env import specs

import chex
import jax
import rlax
from rlax._src import distributions
import tjax

from imprl import mdp


class Policy(Protocol):

    def sample(self, key, observations):
        raise NotImplementedError

    def sample_and_split(self, key, observations):
        key, a_key = jax.random.split(key)
        return key, self.sample(a_key, observations)


class DiscretePolicy(Policy, Protocol):
    action_distribution: distributions.DiscreteDistribution

    def probs(self, observations):
        raise NotImplementedError


class EpsilonGreedyPolicy(DiscretePolicy):

    def __init__(self, epsilon):
        self.action_distribution = rlax.epsilon_greedy(epsilon)

    def preferences(self, observations):
        raise NotImplementedError

    def sample(self, key, observations):
        return self.action_distribution.sample(key, self.preferences(observations))

    def probs(self, observations):
        return self.action_distribution.probs(self.preferences(observations))

    @functools.partial(jax.jit, static_argnums=0)
    def sample_and_split(self, key, observations):
        return super().sample_and_split(key, observations)


def generate_trajectory(key: chex.PRNGKey,
                        env: dm_env.Environment,
                        policy: Policy,
                        max_steps: Optional[int] = None,
                        ) -> Tuple[Sequence[dm_env.TimeStep], Sequence]:
    t = 0
    timestep = env.reset()
    trajectory = [timestep]
    actions = []

    while (not timestep.last()) and (max_steps is None or t < max_steps):
        key, action = policy.sample_and_split(key, timestep.observation)
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
                                       reward_offset: Optional[float] = None,
                                       ) -> Tuple[np.ndarray, np.ndarray]:
    if reward_offset is None:
        reward_offset = 0.

    _, rewards, discounts, observations = zip(*trajectory)
    returns = [0.]
    for rew, discount in zip(rewards[:0:-1], discounts[:0:-1]):
        returns.append(rew + reward_offset + extra_discount * discount * returns[-1])
    return np.array(observations[:-1]), np.array(returns[:0:-1])


@gin.register
class MDPEnv(dm_env.Environment):
    """A stateful dm_env wrapper for an MDP.
    """

    def __init__(self, *,
                 seed: int,
                 mdp: mdp.DenseMDP,
                 init_prob: Optional[chex.Array] = None):
        self._mdp = mdp
        self._init_prob = init_prob
        self._key = jax.random.PRNGKey(seed)
        self._state = None

    def reset(self):
        reset_key, self._key = jax.random.split(self._key)
        self._state = jax.random.choice(reset_key, self._mdp.num_states(), p=self._init_prob)
        return dm_env.restart(self._state)

    def step(self, action):
        step_key, self._key = jax.random.split(self._key)
        reward = self._mdp.rewards[self._state, action]
        self._state = self._mdp.sample_next_state(step_key, self._state, action)

        return dm_env.transition(reward=reward, observation=self._state)

    def observation_spec(self):
        return specs.DiscreteArray(self._mdp.num_states(), name='state')

    def action_spec(self):
        return specs.DiscreteArray(self._mdp.num_actions(), name='action')
