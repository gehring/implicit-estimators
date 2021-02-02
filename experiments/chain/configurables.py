import gin
import numpy as np

import chex
import rlax

import jax

from imprl import mdp

from experiments import rollout


@gin.configurable
def mdp_data_factory(key, data):
    del key
    # Since we're learning values for all MDP states, from a supervised learning perspective, the
    # train and test data are the same.
    return data, data


@gin.configurable
class ChainPolicy(rollout.EpsilonGreedyPolicy):

    def preferences(self, observations):
        del observations
        return np.array([1., 0.])


@gin.configurable
def create_chain_mdp(num_states, slip_prob=0., good_reward=10., bad_reward=1.,
                     discount=0.99):
    next_trans = np.roll(np.eye(num_states), 1, axis=-1)
    next_trans[-2, :] = next_trans[-1, :]
    next_trans = slip_prob * np.eye(num_states) + (1 - slip_prob) * next_trans

    reset_trans = np.zeros((num_states, num_states))
    reset_trans[:-1, -1] = 1.
    reset_trans[-1, 0] = 1.
    reset_trans = slip_prob * np.eye(num_states) + (1 - slip_prob) * reset_trans

    next_rew = np.zeros(num_states)
    next_rew[-2] = good_reward

    reset_rew = np.zeros(num_states)
    reset_rew[-1] = bad_reward
    return mdp.DenseMDP(
        np.stack((next_rew, reset_rew), axis=1),
        mdp.DenseProbs(np.stack((next_trans, reset_trans), axis=1)),
        discount,
    )
