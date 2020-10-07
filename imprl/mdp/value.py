from typing import Any, Callable, Protocol, TypeVar

import jax
import jax.numpy as jnp
import tjax
from jax.scipy import special

from imprl.mdp.base import MDP, Array, StateActionArray

T = TypeVar("T", bound=Any)
S = TypeVar("S", bound=Any)
A = TypeVar("A", bound=Any)


def identity_offset(x: Array[T, S]) -> Array[T, S]:
    return x


def mean_offset(x: Array[T, S]) -> Array[T, S]:
    return x - x.mean()


def max_reduce(x: StateActionArray[T, S, A]) -> Array[T, S]:
    return x.max(axis=-1)


def logsumexp_reduce(x: StateActionArray[T, S, A]) -> Array[T, S]:
    return special.logsumexp(x, axis=-1)


class ValueSolver(Protocol[T, S, A]):
    tol: float
    maxiter: int

    def reduce(self, qvalues: StateActionArray[T, S, A]) -> Array[T, S]:
        raise NotImplementedError

    def offset(self, values: Array[T, S]) -> Array[T, S]:
        raise NotImplementedError

    def __call__(self, init_values: Array[T, S], mdp: MDP[T, S, A]):
        raise NotImplementedError

    def update_values(self, mdp: MDP[T, S, A], values: Array[T, S]) -> Array[T, S]:
        qvalues = mdp.lookahead_qvalues(values)
        return self.offset(self.reduce(qvalues))


@tjax.dataclass
class ValueIteration(ValueSolver[T, S, A]):
    tol: float = tjax.field(static=True)  # type: ignore
    maxiter: int = tjax.field(static=True)  # type: ignore
    reduce: Callable[[StateActionArray], Array] = tjax.field(default=max_reduce, static=True)  # type: ignore  # noqa: E501
    offset: Callable[[Array], Array] = tjax.field(default=identity_offset, static=True)  # type: ignore  # noqa: E501

    def __post_init__(self):
        if self.reduce is None:
            super().__setattr__("reduce", max_reduce)
        if self.offset is None:
            super().__setattr__("offset", identity_offset)

    def __call__(self, init_values: Array[T, S], mdp: MDP[T, S, A]):
        tol_sqr = self.tol * self.tol

        def cond(args):
            values, old_values, i = args
            diff = values - old_values
            max_sqr_diff = (diff * diff).max()
            return jnp.logical_or(max_sqr_diff > tol_sqr, i >= self.maxiter)

        def body(args):
            values, _, i = args
            new_values = self.update_values(mdp, values)
            return new_values, values, i + 1

        new_values = self.update_values(mdp, init_values)
        optimal_values, *_ = jax.lax.while_loop(
            cond,
            body,
            (new_values, init_values, 1),
        )

        return optimal_values
