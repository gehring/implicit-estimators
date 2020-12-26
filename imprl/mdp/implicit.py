from typing import Any, Callable, Optional, TypeVar

import jax
from jax.scipy.sparse import linalg

import fax.implicit

from imprl.mdp.base import MDP, Array
from imprl.mdp.value import ValueSolver

T = TypeVar("T", bound=Any)
S = TypeVar("S", bound=Any)
A = TypeVar("A", bound=Any)


def normal_equation_solve(A, b, linear_solver=None, tol=1e-3, damp=1e-3,
                          symmetric=False):
  if symmetric is True:
    A_transpose = A
  else:
    transposed_fun = jax.linear_transpose(A, b)
    def A_transpose(x):
      (y,) = transposed_fun(x)
      return y

  if linear_solver is None:
    linear_solver = linalg.gmres

  def _normal_eq(x):
    return jax.tree_multimap(lambda v, u: v + damp * u,
                             A_transpose(A(x)), x)

  return linear_solver(_normal_eq, A_transpose(b), tol=tol)


def bellman_solve(init_values: Array[T, S],
                  mdp: MDP[T, S, A],
                  solver: ValueSolver[T, S, A],
                  rev_solver: Optional[Callable] = None,
                  ) -> Array[T, S]:

    # return fax.implicit.two_phase_solve(
    #     lambda params: lambda x: solver.update_values(params, x),
    #     init_values,
    #     mdp,
    # )
    # if rev_solver is None:
    #     rev_solver = normal_equation_solve

    def bellman_diff(values: Array[T, S], mdp: MDP[T, S, A]) -> Array[T, S]:
        return solver.update_values(mdp, values) - values

    return fax.implicit.root_solve(bellman_diff, init_values, mdp, solver, rev_solver=rev_solver)
