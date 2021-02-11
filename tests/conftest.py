from functools import partial
from typing import NamedTuple

import pytest

import jax

import numpy as np

from imprl.mdp import value, base, dense


class MDPTestInstance(NamedTuple):
    mdp: base.MDP
    values: base.Array


def simple_dense_mdp():
    rewards = np.array([[1., 1.], [0., 0.]])
    probs = np.stack([np.eye(2), 0.5 * np.ones((2, 2))], axis=1)
    discounts = 0.99
    mdp = dense.DenseMDP(rewards, dense.DenseProbs(probs), discounts)
    values = np.array([100., discounts**2 * 100.])
    return MDPTestInstance(mdp, values)


@pytest.fixture(scope="session", params=[simple_dense_mdp()])
def mdp_instance(request, value_solver):
    mdp, values = request.param
    return MDPTestInstance(mdp, value_solver.offset(values))


@pytest.fixture(scope="session", params=[partial(value.ValueIteration, tol=1e-7, maxiter=2000)])
def solver_cls(request):
    return request.param


@pytest.fixture(scope="session", params=[
    value.max_reduce,
    None,
])
def solver_reduce(request):
    return request.param


@pytest.fixture(scope="session", params=[
    value.mean_offset,
    None,
])
def solver_offset(request):
    return request.param


@pytest.fixture(scope="session")
def value_solver(solver_cls, solver_reduce, solver_offset):
    solver = solver_cls(reduce=solver_reduce, offset=solver_offset)
    return solver
