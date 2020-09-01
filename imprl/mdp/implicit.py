from typing import Any, Callable, Optional, TypeVar

import fax.implicit

from imprl.mdp.base import MDP, Array
from imprl.mdp.value import ValueSolver

T = TypeVar("T", bound=Any)
S = TypeVar("S", bound=Any)
A = TypeVar("A", bound=Any)


def bellman_solve(init_values: Array[T, S],
                  mdp: MDP[T, S, A],
                  solver: ValueSolver[T, S, A],
                  rev_solver: Optional[Callable] = None,
                  ) -> Array[T, S]:

    def bellman_diff(values: Array[T, S], mdp: MDP[T, S, A]) -> Array[T, S]:
        return solver.update_values(mdp, values)[0] - values

    return fax.implicit.root_solve(bellman_diff, init_values, mdp, solver, rev_solver=rev_solver)
