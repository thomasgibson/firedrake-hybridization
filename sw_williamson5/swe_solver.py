from __future__ import absolute_import, print_function, division

from firedrake import (LinearVariationalProblem, LinearVariationalSolver,
                       TestFunctions, TrialFunctions, Function,
                       lhs, rhs, inner, div, dx, split)

from gusto import TimesteppingSolver as Solver


class SWESolver(Solver):

    def _setup_solver(self):

        state = self.state
        H = state.parameters.H
        g = state.parameters.g
        beta = state.timestepping.dt*state.timestepping.alpha

        # Split rhs vector symbolically
        u_in, D_in = split(state.xrhs)

        W = state.W
        w, phi = TestFunctions(W)
        u, D = TrialFunctions(W)

        eqn = (inner(w, u) - beta*g*div(w)*D - inner(w, u_in)
               + phi*D + beta*H*phi*div(u) - phi*D_in)*dx

        a = lhs(eqn)
        L = rhs(eqn)

        # Place to put result of mixed solution
        self.uD = Function(W)

        # Solver for (u, D)
        uD_problem = LinearVariationalProblem(a, L, self.state.dy)

        self.uD_solver = LinearVariationalSolver(uD_problem,
                                                 solver_parameters=self.params,
                                                 options_prefix="SWE-implicit")

    def solve(self):
        self.uD_solver.solve()
