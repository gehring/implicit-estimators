from typing import Callable, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import tjax
from tjax.annotations import PyTree, RealArray

from imprl.mdp.dense import DenseLogits, DenseMDP
from imprl.modules.base import Module


def init_log_transitions(key, num_states, num_actions, dtype):
    # Change this if you want a different distribution of inital weights.
    shape = (num_states, num_actions, num_states)
    logprob = jnp.log(
        jax.random.dirichlet(key, jnp.ones(shape), dtype=dtype)
        + 2*jnp.eye(shape[2])[:, None, :]
    )  # + 3 * jnp.eye(shape[2])[:, None, :]
    return logprob - jnp.max(logprob, axis=-1)[..., None]
    # logprob = jnp.log(jax.random.dirichlet(key, jnp.ones(shape), dtype=dtype) + jnp.eye(shape[2])[:, None, :]) # + 3 * jnp.eye(shape[2])[:, None, :]
    # return logprob - jnp.max(logprob, axis=-1)[..., None]
    # return (2 * jnp.eye(shape[2])[:, None, :]
    #         + jnp.exp(0.1 * jax.random.normal(key, shape, dtype=dtype)))


def init_rewards(key, num_states, num_actions, dtype):
    # Change this if you want a different distribution of inital weights.
    shape = (num_states, num_actions)
    return jax.random.normal(key, shape, dtype=dtype) * 0.1


class MDPParams(NamedTuple):
    rewards: PyTree
    transitions: PyTree
    discounts: Optional[PyTree] = None


@tjax.dataclass
class ExplicitMDP(Module[DenseMDP]):
    num_pseudo_actions: int
    rewards_init: Callable = tjax.field(default=init_rewards, static=True)  # type: ignore
    log_transition_init: Callable = tjax.field(default=init_log_transitions, static=True)  # type: ignore  # noqa: E501
    discount_init: Union[Callable, RealArray] = tjax.field(static=True)

    def init(self, rng, inputs):
        num_states = inputs.shape[-1]
        rew_key, trans_key, discount_key = jax.random.split(rng, 3)

        log_transitions = self.log_transition_init(
            trans_key, num_states, self.num_pseudo_actions, inputs.dtype)
        rewards = self.rewards_init(
            rew_key, num_states, self.num_pseudo_actions, inputs.dtype)

        discounts = None
        if callable(self.discount_init):
            discounts = self.discount_init(
                discount_key, num_states, self.num_pseudo_actions, inputs.dtype)
        #     rewards / (1 - discounts)
        # else:
        #     rewards / (1 - self.discount_init)

        return MDPParams(rewards, log_transitions, discounts)

    def apply(self, params, *args):
        rewards, log_transitions, discounts = params

        if discounts is None:
            discounts = self.discount_init

        return DenseMDP(
            rewards,
            # rewards * (1 - discounts),
            DenseLogits(log_transitions),
            discounts
        )
