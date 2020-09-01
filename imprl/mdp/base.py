import abc
import operator
from typing import (Any, Hashable, Literal, Optional, Protocol, Sequence, Tuple, Type, TypeVar,
                    overload, runtime_checkable)

import jax.tree_util
from chex import PRNGKey
from tjax.annotations import PyTree
from tjax.pytree_like import PyTreeLike

T = TypeVar("T", bound=Any)
S = TypeVar("S", bound=Any)
A = TypeVar("A", bound=Any)


class PyTreeArray(PyTreeLike):

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        jax.tree_util.register_pytree_node_class(cls)

    @abc.abstractmethod
    def tree_flatten(self) -> Tuple[Sequence[PyTree], Hashable]:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def tree_unflatten(cls: Type[T], hashed: Hashable, trees: Sequence[PyTree]) -> T:
        raise NotImplementedError


@runtime_checkable
class Array(Protocol[T, S]):

    def __getitem__(self, key: S) -> T:
        raise NotImplementedError

    def __add__(self, other: "Array[T, S]") -> "Array[T, S]":
        return jax.tree_multimap(operator.add, self, other)

    def __sub__(self, other: "Array[T, S]") -> "Array[T, S]":
        return jax.tree_multimap(operator.sub, self, other)

    def __mul__(self, other: "Array[T, S]") -> "Array[T, S]":
        return jax.tree_multimap(operator.mul, self, other)

    def __truediv__(self, other: "Array[T, S]") -> "Array[T, S]":
        return jax.tree_multimap(operator.truediv, self, other)

    def mean(self) -> T:
        return self.sum() / self.size

    def sum(self) -> T:
        return sum(x.sum() for x in jax.tree_leaves(self))  # type: ignore

    @property
    def size(self) -> int:
        return sum(x.size for x in jax.tree_leaves(self))


@runtime_checkable
class StateActionArray(Protocol[T, S, A]):

    def __getitem__(self, key: Tuple[S, A]) -> T:
        raise NotImplementedError

    def __add__(self, other: "StateActionArray[T, S, A]") -> "StateActionArray[T, S, A]":
        return jax.tree_multimap(operator.add, self, other)

    def __sub__(self, other: "StateActionArray[T, S, A]") -> "StateActionArray[T, S, A]":
        return jax.tree_multimap(operator.sub, self, other)

    def __mul__(self, other: "StateActionArray[T, S, A]") -> "StateActionArray[T, S, A]":
        return jax.tree_multimap(operator.mul, self, other)

    def __truediv__(self, other: "StateActionArray[T, S, A]") -> "StateActionArray[T, S, A]":
        return jax.tree_multimap(operator.truediv, self, other)

    @overload
    def argmax(self, axis: Literal[1]) -> A:
        ...

    @overload
    def argmax(self, axis: Literal[0]) -> S:
        ...

    @overload
    def argmax(self, axis: None) -> Tuple[S, A]:
        ...

    def argmax(self, axis: Optional[Literal[0, 1]] = None) -> Any:
        raise NotImplementedError

    @overload
    def max(self, axis: Literal[1, -1]) -> Array[T, S]:
        ...

    @overload
    def max(self, axis: Literal[0, -2]) -> Array[T, A]:
        ...

    @overload
    def max(self, axis: None) -> T:
        ...

    def max(self, axis: Optional[Literal[-2, -1, 0, 1]] = None) -> Any:
        raise NotImplementedError


class ConditionalDistribution(Protocol[S, A]):

    def sample(self, rng: PRNGKey, state: S, action: A) -> S:
        raise NotImplementedError

    @overload
    def expectation(self, values: Array[T, S]) -> StateActionArray[T, S, A]:
        ...

    @overload
    def expectation(self, values: Array[T, S], state: S, action: A) -> T:
        ...

    def expectation(self,
                    values: Array[T, S],
                    state: Optional[S] = None,
                    action: Optional[A] = None) -> Any:
        raise NotImplementedError


class MDP(Protocol[T, S, A]):
    rewards: StateActionArray[T, S, A]
    transitions: ConditionalDistribution[S, A]
    discounts: StateActionArray[T, S, A]

    def lookahead_qvalues(self, values: Array[T, S]) -> StateActionArray[T, S, A]:
        return self.rewards + self.discounts * self.transitions.expectation(values)

    def sample_next_state(self, rng: PRNGKey, state: S, action: A) -> S:
        return self.transitions.sample(rng, state, action)
