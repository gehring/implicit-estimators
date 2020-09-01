import numpy as np

import jax
import jax.numpy as jnp

from imprl.mdp import dense


def test_dense_mdp():
    rewards = jnp.array([[1., 1.], [0., 0.]])
    probs = jnp.stack([jnp.eye(2), 0.5 * jnp.ones((2, 2))], axis=1)
    mdp = dense.DenseMDP(rewards, dense.DenseProbs(probs), 1.)

    @jax.vmap
    def sample_next(rng, state, action):
        return mdp.sample_next_state(rng, state, action)

    k = 10000
    rng = jax.random.PRNGKey(0)
    states = jnp.array([0] * k + [1] * k, dtype=jnp.int32)

    # check invariance of action 0
    next_states = sample_next(jax.random.split(rng, k * 2), states, jnp.zeros((k * 2,), jnp.int32))
    np.testing.assert_allclose(next_states, states)

    # check 1st moment of action 1
    next_states = sample_next(jax.random.split(rng, k * 2), states, jnp.ones((k * 2,), jnp.int32))
    np.testing.assert_allclose(jnp.mean(next_states[:k]), 0.5, atol=1e-2)
    np.testing.assert_allclose(jnp.mean(next_states[k:]), 0.5, atol=1e-2)

    # check the one and two step qvalues
    one_step_q = mdp.lookahead_qvalues(jnp.zeros((2,)))
    two_step_q = mdp.lookahead_qvalues(one_step_q.max(-1))
    np.testing.assert_allclose(one_step_q, rewards)
    np.testing.assert_allclose(two_step_q, [[2., 1.5], [0., 0.5]])
