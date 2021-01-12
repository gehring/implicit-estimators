import itertools
from typing import AbstractSet, Tuple

import gin

import numpy as np

from imprl.mdp import dense


_GRID_ACTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


@gin.configurable
def mdp_data_factory(key, data):
    del key
    # Since we're learning values for all MDP states, from a supervised learning perspective, the
    # train and test data are the same.
    return data, data


def _inside_grid(x, size):
    nonneg = all(i >= 0 for i in x)
    return nonneg and all(i < ub for i, ub in zip(x, size))


def dense_gridworld(size: Tuple[int, int],
                    goal: Tuple[int, int],
                    discount: float,
                    obstacles: AbstractSet[Tuple[int, int]]) -> dense.DenseMDP:
    if not _inside_grid(goal, size):
        raise ValueError((
            "The goal must be non-negative and can't be outside the max size of the gridworld. "
            f"Received size={size} and goal={goal}"
        ))
    if not (0 <= discount <= 1):
        raise ValueError(f"Discount must be within 0 and 1 but discount={discount}")

    n = np.prod(size)
    rewards = np.zeros((n + 1, len((_GRID_ACTIONS))))
    probs = np.zeros((n + 1, len(_GRID_ACTIONS), n + 1))

    # set last state as a terminal absorbing state
    probs[-1, :, -1] = 1.

    for x in itertools.product(*(range(s) for s in size)):
        ix = np.ravel_multi_index(x, size)
        if x == goal:
            # goal state always moves to terminal state with a reward of 1.0
            rewards[ix] = 1.
            probs[ix, :, -1] = 1.
        else:
            for a, dx in enumerate(_GRID_ACTIONS):
                next_x = tuple(np.add(x, dx))
                if not _inside_grid(next_x, size) or next_x in obstacles:
                    # if illegal move, don't move
                    next_ix = ix
                else:
                    next_ix = np.ravel_multi_index(next_x, size)
                probs[ix, a, next_ix] = 1.

    return dense.DenseMDP(rewards, dense.DenseProbs(probs), discount)


_FOUR_ROOM_SIZE = (11, 11)
_FOUR_ROOM_OBSTACLES = {
    (0, 5), (2, 5), (3, 5), (4, 5),
    (6, 4), (7, 4), (9, 4), (10, 4),
    (5, 0), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 9), (5, 10),
}


@gin.configurable
def create_four_rooms(goal, discount, fail_prob=1/3):
    # The four rooms MDP as described in Sutton et al [1].
    #
    # [1] Sutton, Richard S., Doina Precup, and Satinder Singh. "Between MDPs and semi-MDPs: A
    # framework for temporal abstraction in reinforcement learning." Artificial intelligence
    # 112.1 - 2(1999): 181 - 211.
    if not (0 <= fail_prob <= 1):
        raise ValueError(f"`fail_prob` must be between 0 and 1 but received {fail_prob}")

    mdp = dense_gridworld(
        size=_FOUR_ROOM_SIZE,
        goal=goal,
        discount=discount,
        obstacles=_FOUR_ROOM_OBSTACLES,
    )
    probs = mdp.transitions[:, :, :]
    m = probs.shape[1]

    # Ratio of different action transitions where failing results in a executing a different
    # action uniformly at random.
    scales = np.eye(m) * (1 - fail_prob) + fail_prob * (1 - np.eye(m))/(m - 1)
    rewards = np.einsum("ia,ba->ib", mdp.rewards, scales)
    probs = np.einsum("iaj,ba->ibj", probs, scales)

    assert np.allclose(probs.sum(axis=-1), 1)
    probs /= probs.sum(axis=-1)[..., None]

    return dense.DenseMDP(rewards, dense.DenseProbs(probs), mdp.discounts)
