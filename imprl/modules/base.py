from typing import Any, Protocol, TypeVar

from chex import PRNGKey
from tjax.annotations import PyTree

T = TypeVar("T", bound=Any, covariant=True)


class Module(Protocol[T]):

    def init(self, rng: PRNGKey, *args: PyTree) -> PyTree:
        raise NotImplementedError

    def apply(self, params: PyTree, *args: PyTree) -> T:
        raise NotImplementedError
