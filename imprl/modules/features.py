from collections import abc
import itertools
from typing import Any, Protocol, TypeVar

import tjax

import numpy as np

from tjax.annotations import PyTree

import jax
import jax.numpy as jnp

import dm_env

T = TypeVar("T", bound=Any, covariant=True)


class Encoder(Protocol[T]):

    def apply(self, observation: PyTree) -> T:
        raise NotImplementedError


def _as_iterable(bound, max_repeat=None):
    if not isinstance(bound, abc.Iterable):
        bound = itertools.repeat(bound, max_repeat)
    return bound


def uniform_centers(env: dm_env.Environment, centers_per_dim: int):
    obs_spec = env.observation_spec()
    assert len(obs_spec.shape) == 1, "Only rank 1 observations are supported."

    minimums = _as_iterable(obs_spec.minimum, obs_spec.shape[0])
    maximums = _as_iterable(obs_spec.maximum, obs_spec.shape[0])

    centers = np.meshgrid(*[
        np.linspace(lb, ub, centers_per_dim, endpoint=True)
        for lb, ub in zip(minimums, maximums)
    ])
    centers = np.stack([x.flatten() for x in centers], axis=0)

    return centers


def normalized_scales(env: dm_env.Environment, scale: float):
    obs_spec = env.observation_spec()
    assert len(obs_spec.shape) == 1, "Only rank 1 observations are supported."

    minimums = list(_as_iterable(obs_spec.minimum, obs_spec.shape[0]))
    maximums = list(_as_iterable(obs_spec.maximum, obs_spec.shape[0]))
    span = np.subtract(maximums, minimums)

    return span * scale


@tjax.dataclass
class RBFEncoder(Encoder):
    """ A feature encoding using radial basis functions.
    """
    centers: bool = tjax.field(static=True)  # type: ignore
    scales: bool = tjax.field(static=True)  # type: ignore
    normalized: bool = tjax.field(static=True)  # type: ignore

    def apply(self, inputs):
        diff = (inputs[..., None] - self.centers) / self.scales[..., None]
        neg_dist = -jnp.sum(diff**2, axis=-2)
        if self.normalized:
            return jax.nn.softmax(neg_dist)
        else:
            return jnp.exp(neg_dist)


@tjax.dataclass
class OneHot(Encoder):
    """ A one-hot encoder.
    """
    dim: int = tjax.field(static=True)  # type: ignore

    def apply(self, inputs):
        return jax.nn.one_hot(inputs, self.dim)
