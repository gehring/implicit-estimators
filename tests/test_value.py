import chex

import jax
import jax.numpy as jnp


def test_value_solver(value_solver, mdp_instance):
    mdp, true_values = mdp_instance

    values = value_solver(jax.tree_map(jnp.zeros_like, true_values), mdp)
    chex.assert_tree_all_close(
        values,
        true_values,
        atol=1e-2,
        rtol=1e-4,
    )

    # assert solution is a fixed-point.
    chex.assert_tree_all_close(
        value_solver(values, mdp),
        values
    )

    # assert that no 1-step qvalue is larger than the state value
    updated_values = mdp.lookahead_qvalues(values).max(axis=1)
    updated_values, _ = value_solver.offset(updated_values)
    no_improve = jax.tree_multimap(lambda x, y: jnp.all(x <= y), updated_values, values)
    assert all(jax.tree_leaves(no_improve))
