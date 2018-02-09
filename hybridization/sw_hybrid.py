from firedrake import split, LinearVariationalProblem, \
    LinearVariationalSolver, TestFunctions, TrialFunctions, \
    lhs, rhs, div, dx, inner, cross, Function, CellNormal

from gusto.linear_solvers import TimesteppingSolver


__all__ = ["HybridizedShallowWaterSolver"]


class HybridizedShallowWaterSolver(TimesteppingSolver):
    """
    Timestepping linear solver object for the nonlinear shallow water
    equations with prognostic variables u and D. This linear solver
    includes the Coriolis term in the linearized equations.

    This solver uses the HybridizationPC preconditioner from the core
    Firedrake package.
    """

    solver_parameters = {
        'ksp_type': 'preonly',
        'mat_type': 'matfree',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.HybridizationPC',
        'hybridization': {'ksp_type': 'cg',
                          'pc_type': 'gamg',
                          'ksp_rtol': 1e-8,
                          'mg_levels': {'ksp_type': 'chebyshev',
                                        'ksp_max_it': 2,
                                        'pc_type': 'bjacobi',
                                        'sub_pc_type': 'ilu'}}
    }

    def _setup_solver(self):
        state = self.state
        H = state.parameters.H
        g = state.parameters.g
        beta = state.timestepping.dt*state.timestepping.alpha
        f = state.fields("coriolis")

        # Split up the rhs vector (symbolically)
        u_in, D_in = split(state.xrhs)

        W = state.W
        w, phi = TestFunctions(W)
        u, D = TrialFunctions(W)

        outward_normals = CellNormal(state.mesh)

        def perp(arg):
            return cross(outward_normals, arg)

        eqn = (
            inner(w, u) - beta*g*div(w)*D
            - beta*inner(w, f*perp(u))
            - inner(w, u_in)
            + phi*D + beta*H*phi*div(u)
            - phi*D_in
        )*dx

        aeqn = lhs(eqn)
        Leqn = rhs(eqn)

        # Place to put result of u rho solver
        self.uD = Function(W)

        # Solver for u, D
        uD_problem = LinearVariationalProblem(aeqn, Leqn, self.state.dy)

        self.uD_solver = LinearVariationalSolver(uD_problem,
                                                 solver_parameters=self.solver_parameters,
                                                 options_prefix='SWimplicit')

    def solve(self):
        """
        Apply the solver with rhs state.xrhs and result state.dy.
        """

        self.uD_solver.solve()
