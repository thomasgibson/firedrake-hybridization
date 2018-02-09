from firedrake import split, LinearVariationalProblem, \
    LinearVariationalSolver, TestFunctions, TrialFunctions, \
    TestFunction, TrialFunction, lhs, rhs, FacetNormal, \
    div, dx, jump, avg, dS_v, dS_h, ds_v, ds_t, ds_b, inner, dot, grad, \
    MixedFunctionSpace, FunctionSpace, Function, \
    BrokenElement, assemble, LinearSolver, Tensor, AssembledVector

from firedrake.parloops import par_loop, READ, INC

from gusto.linear_solvers import TimesteppingSolver
from gusto.forcing import exner, exner_rho, exner_theta


__all__ = ["HybridizedCompressibleSolver"]


class HybridizedCompressibleSolver(TimesteppingSolver):
    """
    Timestepping linear solver object for the compressible equations
    in theta-pi formulation with prognostic variables u, rho, and theta.

    This solver follows the following strategy:

    (1) Analytically eliminate theta (introduces error near topography)

    (2a) Formulate the resulting mixed system for u and rho using a
         hybridized mixed method. This breaks continuity in the
         linear perturbations of u, and introduces a new unknown on the
         mesh interfaces approximating the average of the Exner pressure
         perturbations. These trace unknowns also act as Lagrange
         multipliers enforcing normal continuity of the "broken" u variable.

    (2b) Statically condense the block-sparse system into a single system
         for the Lagrange multipliers. This is the only globally coupled
         system requiring a linear solver.

    (2c) Using the computed trace variables, we locally recover the
         broken velocity and density perturbations. This is accomplished
         in two stages:
         (i): Recover rho locally using the multipliers.
         (ii): Recover "broken" u locally using rho and the multipliers.

    (2d) Project the "broken" velocity field into the HDiv-conforming
         space using local averaging.

    (3) Reconstruct theta

    :arg state: a :class:`.State` object containing everything else.
    :arg quadrature degree: tuple (q_h, q_v) where q_h is the required
    quadrature degree in the horizontal direction and q_v is that in
    the vertical direction
    :arg solver_parameters (optional): solver parameters for the
    trace system
    :arg overwrite_solver_parameters: boolean, if True use only the
    solver_parameters that have been passed in, if False then update
    the default solver parameters with the solver_parameters passed in.
    :arg moisture (optional): list of names of moisture fields.
    """

    # Solver parameters for the Lagrange multiplier system
    # NOTE: The reduced operator is not symmetric
    solver_parameters = {'ksp_type': 'gmres',
                         'pc_type': 'gamg',
                         'mg_levels': {'ksp_type': 'chebyshev',
                                       'ksp_chebyshev_esteig': True,
                                       'ksp_max_it': 1,
                                       'pc_type': 'bjacobi',
                                       'sub_pc_type': 'ilu'}}

    def __init__(self, state, quadrature_degree=None, solver_parameters=None,
                 overwrite_solver_parameters=False, moisture=None):

        self.moisture = moisture

        self.state = state

        if quadrature_degree is not None:
            self.quadrature_degree = quadrature_degree
        else:
            dgspace = state.spaces("DG")
            if any(deg > 2 for deg in dgspace.ufl_element().degree()):
                state.logger.warning("default quadrature degree most likely not sufficient for this degree element")
            self.quadrature_degree = (5, 5)

        super().__init__(state, solver_parameters, overwrite_solver_parameters)

    def _setup_solver(self):
        from firedrake.assemble import create_assembly_callable
        import numpy as np

        state = self.state
        dt = state.timestepping.dt
        beta = dt*state.timestepping.alpha
        cp = state.parameters.cp
        mu = state.mu
        Vu = state.spaces("HDiv")
        Vu_broken = FunctionSpace(state.mesh, BrokenElement(Vu.ufl_element()))
        Vtheta = state.spaces("HDiv_v")
        Vrho = state.spaces("DG")

        h_deg = state.horizontal_degree
        v_deg = state.vertical_degree
        Vtrace = FunctionSpace(state.mesh, "HDiv Trace", degree=(h_deg, v_deg))

        # Split up the rhs vector (symbolically)
        u_in, rho_in, theta_in = split(state.xrhs)

        # Build the reduced function space for "broken" u and rho
        M = MixedFunctionSpace((Vu_broken, Vrho))
        w, phi = TestFunctions(M)
        u, rho = TrialFunctions(M)

        # Introduce test and trial functions on the trace space
        l0 = TrialFunction(Vtrace)

        n = FacetNormal(state.mesh)

        # Get background fields
        thetabar = state.fields("thetabar")
        rhobar = state.fields("rhobar")
        pibar = exner(thetabar, rhobar, state)
        pibar_rho = exner_rho(thetabar, rhobar, state)
        pibar_theta = exner_theta(thetabar, rhobar, state)

        # Analytical (approximate) elimination of theta
        k = state.k             # Upward pointing unit vector
        theta = -dot(k, u)*dot(k, grad(thetabar))*beta + theta_in

        # Only include theta' (rather than pi') in the vertical
        # component of the gradient

        # The pi prime term (here, bars are for mean and no bars are
        # for linear perturbations)
        pi = pibar_theta*theta + pibar_rho*rho

        # Vertical projection
        def V(u):
            return k*inner(u, k)

        # Specify degree for some terms as estimated degree is too large
        dxp = dx(degree=(self.quadrature_degree))
        dS_vp = dS_v(degree=(self.quadrature_degree))
        dS_hp = dS_h(degree=(self.quadrature_degree))
        ds_vp = ds_v(degree=(self.quadrature_degree))
        ds_tbp = ds_t(degree=(self.quadrature_degree)) + ds_b(degree=(self.quadrature_degree))

        # Add effect of density of water upon theta
        if self.moisture is not None:
            water_t = Function(Vtheta).assign(0.0)
            for water in self.moisture:
                water_t += self.state.fields(water)
            theta = theta / (1 + water_t)
            thetabar = thetabar / (1 + water_t)

        # "broken" u and rho system (minus momentum surface terms)
        Aeqn = (inner(w, (state.h_project(u) - u_in))*dx
                - beta*cp*div(theta*V(w))*pibar*dxp
                # TODO: Need to think about whether we still need this
                # term after breaking the space Vu.
                + beta*cp*jump(theta*V(w), n=n)*avg(pibar)*dS_vp
                - beta*cp*div(thetabar*w)*pi*dxp
                + (phi*(rho - rho_in) - beta*inner(grad(phi), u)*rhobar)*dx
                + beta*jump(phi*u, n=n)*avg(rhobar)*(dS_v + dS_h))

        if mu is not None:
            Aeqn += dt*mu*inner(w, k)*inner(u, k)*dx

        # Form the mixed operators using Slate
        # (A   K)(U) = (U_r)
        # (K.T 0)(l)   (0  )
        Aop = Tensor(lhs(Aeqn))
        Arhs = Tensor(rhs(Aeqn))

        # Off-diagonal block matrices containing the contributions
        # of the Lagrange multipliers (surface terms in the momentum equation)
        K = Tensor(beta*cp*jump(thetabar*w, n=n)*l0('+')*(dS_vp + dS_hp)
                   + beta*cp*inner(thetabar*w, n)*l0*ds_vp
                   + beta*cp*inner(thetabar*w, n)*l0*ds_tbp)

        # U = A.inv * (U_r - K * l),
        # 0 = K.T * U = -(K.T * A.inv * K) * l + K.T * A.inv * U_r,
        # so (K.T * A.inv * K) * l = K.T * A.inv * U_r
        # is the reduced equation for the Lagrange multipliers.
        # Right-hand side expression: (Forward substitution)
        Rexp = K.T * Aop.inv * Arhs
        self.R = Function(Vtrace)

        # We need to rebuild R everytime data changes
        self._assemble_Rexp = create_assembly_callable(Rexp, tensor=self.R)

        # Schur complement operator:
        Smatexp = K.T * Aop.inv * K
        S = assemble(Smatexp)

        # Set up the Linear solver for the system of Lagrange multipliers
        self.lSolver = LinearSolver(S, solver_parameters=self.solver_parameters,
                                    options_prefix='lambda_solve')

        # Result function for the multiplier solution
        self.lambdar = Function(Vtrace)

        # Place to put result of u rho reconstruction
        self.urho = Function(M)

        # Reconstruction of broken u and rho
        u_, rho_ = self.urho.split()

        # Split operators for two-stage reconstruction
        A00 = Aop.block((0, 0))
        A01 = Aop.block((0, 1))
        A10 = Aop.block((1, 0))
        A11 = Aop.block((1, 1))
        K0 = K.block((0, 0))
        Ru = Arhs.block((0,))
        Rrho = Arhs.block((1,))
        lambda_vec = AssembledVector(self.lambdar)

        # rho reconstruction
        Srho = A11 - A10 * A00.inv * A01
        rho_expr = Srho.inv * (Rrho - A10 * A00.inv * (Ru - K0 * lambda_vec))
        self._assemble_rho = create_assembly_callable(rho_expr, tensor=rho_)

        # "broken" u reconstruction
        rho_vec = AssembledVector(rho_)
        u_expr = A00.inv * (Ru - A01 * rho_vec - K0 * lambda_vec)
        self._assemble_u = create_assembly_callable(u_expr, tensor=u_)

        # Project broken u into the HDiv space using facet averaging.
        # Weight function counting the dofs of the HDiv element:
        shapes = (Vu.finat_element.space_dimension(), np.prod(Vu.shape))

        weight_kernel = """
        for (int i=0; i<%d; ++i) {
        for (int j=0; j<%d; ++j) {
        w[i][j] += 1.0;
        }}""" % shapes

        self._weight = Function(Vu)
        par_loop(weight_kernel, dx, {"w": (self._weight, INC)})

        # Averaging kernel
        self._average_kernel = """
        for (int i=0; i<%d; ++i) {
        for (int j=0; j<%d; ++j) {
        vec_out[i][j] += vec_in[i][j]/w[i][j];
        }}""" % shapes

        # HDiv-conforming velocity
        self.u_hdiv = Function(Vu)

        # Reconstruction of theta
        theta = TrialFunction(Vtheta)
        gamma = TestFunction(Vtheta)

        self.theta = Function(Vtheta)
        theta_eqn = gamma*(theta - theta_in +
                           dot(k, self.u_hdiv)*dot(k, grad(thetabar))*beta)*dx

        theta_problem = LinearVariationalProblem(lhs(theta_eqn), rhs(theta_eqn), self.theta)
        self.theta_solver = LinearVariationalSolver(theta_problem,
                                                    solver_parameters={'ksp_type': 'gmres',
                                                                       'pc_type': 'bjacobi',
                                                                       'pc_sub_type': 'ilu'},
                                                    options_prefix='thetabacksubstitution')

    def solve(self):
        """
        Apply the solver with rhs state.xrhs and result state.dy.
        """
        # Assemble the RHS for lambda into self.R
        self._assemble_Rexp()

        # Solve for lambda
        self.lSolver.solve(self.lambdar, self.R)

        # Reconstruct u and rho
        self._assemble_rho()
        self._assemble_u()

        broken_u, rho1 = self.urho.split()
        u1 = self.u_hdiv

        # Project broken_u into the HDiv space
        u1.assign(0.0)
        par_loop(self._average_kernel, dx,
                 {"w": (self._weight, READ),
                  "vec_in": (broken_u, READ),
                  "vec_out": (u1, INC)})

        # Copy back into u and rho cpts of dy
        u, rho, theta = self.state.dy.split()
        u.assign(u1)
        rho.assign(rho1)

        # Reconstruct theta
        self.theta_solver.solve()

        # Copy into theta cpt of dy
        theta.assign(self.theta)
