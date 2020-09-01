from typing import Any, Callable, Optional, TypeVar, Protocol, Tuple, cast

import jax
import jax.numpy as jnp

import tjax

from imprl.mdp.base import MDP, Array, StateActionArray

T = TypeVar("T", bound=Any)
S = TypeVar("S", bound=Any)
A = TypeVar("A", bound=Any)


def identity_offset(x: Array[T, S], current_offset: Optional[T] = None) -> Tuple[Array[T, S], T]:
    return x, cast(T, 0. if current_offset is None else current_offset)


def mean_offset(x: Array[T, S], current_offset: Optional[T] = None) -> Tuple[Array[T, S], T]:
    mean = x.mean()
    offset = mean
    if current_offset is not None:
        offset = offset + current_offset

    return x - mean, offset


def max_reduce(x: StateActionArray[T, S, A]) -> Array[T, S]:
    return x.max(axis=-1)


class ValueSolver(Protocol[T, S, A]):
    tol: float
    maxiter: int

    def reduce(self, qvalues: StateActionArray[T, S, A]) -> Array[T, S]:
        raise NotImplementedError

    def offset(self, values: Array[T, S], current_offset: Optional[T]) -> Tuple[Array[T, S], T]:
        raise NotImplementedError

    def __call__(self, init_values: Array[T, S], mdp: MDP[T, S, A], relative: bool = False):
        raise NotImplementedError

    def update_values(self,
                      mdp: MDP[T, S, A],
                      values: Array[T, S],
                      offset: Optional[T] = None) -> Tuple[Array[T, S], T]:
        qvalues = mdp.lookahead_qvalues(values)
        return self.offset(self.reduce(qvalues), offset)


@tjax.dataclass
class ValueIteration(ValueSolver[T, S, A]):
    tol: float = tjax.field(static=True)  # type: ignore
    maxiter: int = tjax.field(static=True)  # type: ignore
    reduce: Callable[[StateActionArray], Array] = tjax.field(default=max_reduce, static=True)  # type: ignore  # noqa: E501
    offset: Callable[[Array, Optional[T]], Tuple[Array, T]] = tjax.field(default=identity_offset,
                                                                         static=True)  # type: ignore  # noqa: E501

    def __post_init__(self):
        if self.reduce is None:
            super().__setattr__("reduce", max_reduce)
        if self.offset is None:
            super().__setattr__("offset", identity_offset)

    def __call__(self, init_values: Array[T, S], mdp: MDP[T, S, A], relative: bool = True):
        tol_sqr = self.tol * self.tol

        def cond(args):
            values, _, old_values, i = args
            diff = values - old_values
            max_sqr_diff = (diff * diff).max()
            return jnp.logical_or(max_sqr_diff > tol_sqr, i >= self.maxiter)

        def body(args):
            values, offset, _, i = args
            new_values, new_offset = self.update_values(mdp, values, offset)
            return new_values, new_offset, values, i + 1

        new_values, offset = self.update_values(mdp, init_values)
        optimal_values, offset, *_ = jax.lax.while_loop(
            cond,
            body,
            (new_values, offset, init_values, 1),
        )
        if not relative:
            optimal_values = optimal_values + offset

        return optimal_values
