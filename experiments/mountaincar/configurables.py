from typing import Optional

import gin

import dm_env
from dm_env import specs
import rlax

import numpy as np

import chex
import jax
import jax.numpy as jnp

from experiments import rollout


@gin.configurable
def mountaincar_policy(epsilon: chex.Numeric):

    @jax.jit
    def policy(key: chex.PRNGKey, observation: chex.Array):
        # prefer action along the current velocity
        preferences = jax.lax.cond(observation[1] < 0,
                                   lambda _: jnp.array([1., 0., 0.]),
                                   lambda _: jnp.array([0., 0., 1.]),
                                   None)
        return rlax.epsilon_greedy(epsilon).sample(key, preferences)

    return policy


@gin.configurable
def rollout_dataset(key, *,
                    max_traj_length: int,
                    discount: float,
                    num_traj: Optional[int] = None,
                    num_steps: Optional[int] = None):
    if num_traj == num_steps:
        raise TypeError((
            "Either `num_traj` or `num_steps` is required. Providing a value for both is not "
            "supported."
        ))

    env_seed, policy_seed = np.random.SeedSequence(key).spawn(2)
    policy_key = policy_seed.generate_state(2)

    env = MountainCar(seed=env_seed)
    policy = mountaincar_policy()
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
        data.append(rollout.per_observation_discounted_returns(traj, discount))

        traj_count += 1
        step_count += data[-1][0].shape[0]

    observations, returns = zip(*data)
    return np.concatenate(observations), np.concatenate(returns)


@gin.configurable
def mountaincar_data_factory(key, num_train_traj: int, num_test_states: int):
    train_key, test_key = [s.generate_state(2) for s in np.random.SeedSequence(key).spawn(2)]
    train_data = rollout_dataset(key=train_key, num_traj=num_train_traj)
    test_data = rollout_dataset(key=test_key, num_steps=num_test_states)
    return train_data, test_data


@gin.configurable
class MountainCar(dm_env.Environment):
    """Implementation of the Mountain Car domain.

    Moore, Andrew William. "Efficient memory-based learning for robot control." (1990).

    Default parameters use values presented in Example 10.1 by Sutton & Barto (2018):
    Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press,
    2018.

    """

    def __init__(self,
                 seed: Optional[int] = None,
                 min_pos: float = -1.2,
                 max_pos: float = 0.6,
                 min_init_pos: float = -0.6,
                 max_init_pos: float = -0.4,
                 max_speed: float = 0.07,
                 goal_pos: float = 0.5,
                 force: float = 0.001,
                 gravity: float = 0.0025):
        self._min_pos = min_pos
        self._max_pos = max_pos
        self._min_init_pos = min_init_pos
        self._max_init_pos = max_init_pos
        self._max_speed = max_speed
        self._goal_pos = goal_pos
        self._force = force
        self._gravity = gravity

        self._rng = np.random.default_rng(seed)
        self._position = 0.
        self._velocity = 0.

    def _observation(self):
        return np.array([self._position, self._velocity], np.float32)

    def reset(self):
        self._position = self._rng.uniform(self._min_init_pos, self._max_init_pos)
        self._velocity = 0.
        return dm_env.restart(self._observation())

    def step(self, action):
        """Step the environment

        :param action: 0, 1, 2 correspond to actions left, idle, right, respectively.
        :return: the next timestep
        """
        next_vel = (self._velocity
                    + self._force * (action - 1)
                    - self._gravity * np.cos(self._position * 3))
        self._velocity = np.clip(next_vel, -self._max_speed, self._max_speed)

        self._position = np.clip(self._position + next_vel, self._min_pos, self._max_pos)

        reward = -1
        obs = self._observation()

        if self._position >= self._goal_pos:
            return dm_env.termination(reward=reward, observation=obs)

        return dm_env.transition(reward=reward, observation=obs)

    def observation_spec(self):
        return specs.BoundedArray(
            shape=(2,),
            dtype=np.float32,
            minimum=[self._min_pos, -self._max_speed],
            maximum=[self._max_pos, self._max_speed],
        )

    def action_spec(self):
        """Actions  0, 1, 2 correspond to actions left, idle, right, respectively."""
        return specs.DiscreteArray(3, name='action')
