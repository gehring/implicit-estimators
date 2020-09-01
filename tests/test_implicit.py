import chex

import jax
import jax.numpy as jnp
import jax.test_util

from imprl.mdp import implicit


def test_grad(value_solver, mdp_instance):
    mdp, true_values = mdp_instance
    init_values = jax.tree_map(jnp.zeros_like, true_values)

    def solve_mdp(mdp):
        return implicit.bellman_solve(init_values, mdp, value_solver)

    # jax.grad(solve_mdp)(mdp)
    chex.assert_numerical_grads(solve_mdp, [mdp], order=1, modes=["rev"])
