import jax
import jax.numpy as jnp
import tjax

from imprl.mdp.base import MDP, ConditionalDistribution


@tjax.dataclass
class DenseProbs(ConditionalDistribution[int, int]):
    values: jnp.array

    def __getitem__(self, item):
        return self.values[item]

    def sample(self, rng, state, action):
        probs = self[state, action]
        return jax.random.choice(rng, probs.shape[-1], p=probs)

    def expectation(self, values, state=None, action=None):
        state_idx = slice(None) if state is None else state
        action_idx = slice(None) if action is None else action
        probs = self[state_idx, action_idx]
        return probs @ values


@tjax.dataclass
class DenseLogits(DenseProbs):

    def __getitem__(self, item):
        return jax.nn.softmax(super().__getitem__(item), axis=-1)


@tjax.dataclass
class DenseMDP(MDP[jnp.float32, int, int]):
    rewards: jnp.array
    transitions: DenseProbs
    discounts: jnp.array

    @property
    def size(self):
        return sum(x.size if not isinstance(x, (float, int)) else 1 for x in jax.tree_leaves(self))
