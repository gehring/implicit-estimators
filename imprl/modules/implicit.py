import jax.numpy as jnp
import tjax

from imprl.mdp import implicit
from imprl.mdp.base import MDP
from imprl.mdp.value import ValueSolver
from imprl.modules.base import Module
from imprl.modules.mdp import ExplicitMDP


@tjax.dataclass
class ExplicitWeights(Module):
    """ Initialize some weights such that they are equal to the implicit case.
    """
    implicit_module: Module = tjax.field(static=True)  # type: ignore

    def init(self, key, *inputs):
        return self.implicit_module.apply(self.implicit_module.init(key, *inputs), *inputs)

    def apply(self, params, *inputs):
        del inputs
        return params


@tjax.dataclass
class MDPSolveWeights(Module):
    """ Generate some MDP parameters and solve for the values.
    """
    solver: ValueSolver = tjax.field(static=True)  # type: ignore
    mdp_module: Module[MDP] = tjax.field(static=True)  # type: ignore

    def init(self, key, *inputs):
        return self.mdp_module.init(key, *inputs)

    def apply(self, params, *inputs, previous_values=None):
        mdp = self.mdp_module.apply(params, *inputs)
        if previous_values is None:
            previous_values = mdp.rewards[:, 0]
        return implicit.bellman_solve(previous_values, mdp, self.solver)


@tjax.dataclass
class ExplicitMatrixWeights(Module):
    """ Initialize some weights such that they are equal to the implicit case.
    """
    mdp_module: ExplicitMDP = tjax.field(static=True)  # type: ignore

    def init(self, key, *inputs):
        mdp = self.mdp_module.apply(
            self.mdp_module.init(key, *inputs), *inputs)

        transitions = mdp.transitions[:, :, :]
        assert transitions.shape[1] == 1
        ainv = jnp.eye(transitions.shape[-1]) - mdp.discounts * jnp.squeeze(transitions)
        amat = jnp.linalg.inv(ainv)
        return jnp.squeeze(mdp.rewards), amat

    def apply(self, params, *inputs):
        del inputs
        w, amat = params
        return amat @ w
