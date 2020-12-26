import chex

import jax
import jax.numpy as jnp
import jax.test_util

import numpy as np

from imprl.mdp import dense
from imprl.mdp import implicit


def test_grad(value_solver, mdp_instance):
    mdp, true_values = mdp_instance
    init_values = jax.tree_map(np.zeros_like, true_values)

    def solve_mdp(mdp):
        return implicit.bellman_solve(init_values, mdp, value_solver)

    print(f"solved: {solve_mdp(mdp)}")
    print(jax.grad(lambda *args: jnp.sum(solve_mdp(*args)))(mdp))
    # assert False
    chex.assert_numerical_grads(solve_mdp, [mdp], order=1, modes=["rev"])
