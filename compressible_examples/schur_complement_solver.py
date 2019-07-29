from firedrake import (split, LinearVariationalProblem, Constant,
                       LinearVariationalSolver, TestFunctions, TrialFunctions,
                       TestFunction, TrialFunction, lhs, rhs, DirichletBC, FacetNormal,
                       div, dx, jump, avg, dS_v, dS_h, ds_v, ds_t, ds_b, ds_tb, inner,
                       dot, grad, Function, VectorSpaceBasis, BrokenElement,
                       FunctionSpace, MixedFunctionSpace)
from firedrake.petsc import flatten_parameters, PETSc
from firedrake.parloops import par_loop, READ, INC
from pyop2.profiling import timed_function, timed_region

from gusto.linear_solvers import TimesteppingSolver
from gusto.configuration import logger, DEBUG
from gusto import thermodynamics


__all__ = ['OldCompressibleSolver']


class OldCompressibleSolver(TimesteppingSolver):
    """
    Timestepping linear solver object for the compressible equations
    in theta-pi formulation with prognostic variables u,rho,theta.
    This solver follows the following strategy:
    (1) Analytically eliminate theta (introduces error near topography)
    (2) Solve resulting system for (u,rho) using a Schur preconditioner
    (3) Reconstruct theta
    :arg state: a :class:`.State` object containing everything else.
    :arg quadrature degree: tuple (q_h, q_v) where q_h is the required
         quadrature degree in the horizontal direction and q_v is that in
         the vertical direction
    :arg solver_parameters (optional): solver parameters
    :arg overwrite_solver_parameters: boolean, if True use only the
         solver_parameters that have been passed in, if False then update
         the default solver parameters with the solver_parameters passed in.
    :arg moisture (optional): list of names of moisture fields.
    """

    solver_parameters = {
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'schur',
        'ksp_type': 'gcr',
        'ksp_monitor_true_residual': None,
        'ksp_max_it': 100,
        'pc_fieldsplit_schur_fact_type': 'FULL',
        'pc_fieldsplit_schur_precondition': 'selfp',
        'fieldsplit_0': {'ksp_type': 'preonly',
                         'pc_type': 'bjacobi',
                         'sub_pc_type': 'ilu'},
        'fieldsplit_1': {'ksp_type': 'fgmres',
                         'ksp_monitor_true_residual': None,
                         'ksp_rtol': 1.0e-8,
                         'ksp_atol': 1.0e-8,
                         'ksp_max_it': 100,
                         'pc_type': 'gamg',
                         'pc_gamg_sym_graph': None,
                         'mg_levels': {'ksp_type': 'gmres',
                                       'ksp_max_it': 5,
                                       'pc_type': 'bjacobi',
                                       'sub_pc_type': 'ilu'}}
    }

    def __init__(self, state, quadrature_degree=None, solver_parameters=None,
                 overwrite_solver_parameters=False, moisture=None):

        self.moisture = moisture

        if quadrature_degree is not None:
            self.quadrature_degree = quadrature_degree
        else:
            dgspace = state.spaces("DG")
            if any(deg > 2 for deg in dgspace.ufl_element().degree()):
                logger.warning("default quadrature degree most likely not sufficient for this degree element")
            self.quadrature_degree = (5, 5)

        super().__init__(state, solver_parameters, overwrite_solver_parameters)

    @timed_function("Gusto:SolverSetup")
    def _setup_solver(self):
        state = self.state      # just cutting down line length a bit
        Dt = state.timestepping.dt
        beta_ = Dt*state.timestepping.alpha
        cp = state.parameters.cp
        mu = state.mu
        Vu = state.spaces("HDiv")
        Vtheta = state.spaces("HDiv_v")
        Vrho = state.spaces("DG")

        # Store time-stepping coefficients as UFL Constants
        dt = Constant(Dt)
        beta = Constant(beta_)
        beta_cp = Constant(beta_ * cp)

        # Split up the rhs vector (symbolically)
        u_in, rho_in, theta_in = split(state.xrhs)

        # Build the reduced function space for u,rho
        M = MixedFunctionSpace((Vu, Vrho))
        w, phi = TestFunctions(M)
        u, rho = TrialFunctions(M)

        n = FacetNormal(state.mesh)

        # Get background fields
        thetabar = state.fields("thetabar")
        rhobar = state.fields("rhobar")
        pibar = thermodynamics.pi(state.parameters, rhobar, thetabar)
        pibar_rho = thermodynamics.pi_rho(state.parameters, rhobar, thetabar)
        pibar_theta = thermodynamics.pi_theta(state.parameters, rhobar, thetabar)

        # Analytical (approximate) elimination of theta
        k = state.k             # Upward pointing unit vector
        theta = -dot(k, u)*dot(k, grad(thetabar))*beta + theta_in

        # Only include theta' (rather than pi') in the vertical
        # component of the gradient

        # the pi prime term (here, bars are for mean and no bars are
        # for linear perturbations)

        pi = pibar_theta*theta + pibar_rho*rho

        # vertical projection
        def V(u):
            return k*inner(u, k)

        # specify degree for some terms as estimated degree is too large
        dxp = dx(degree=(self.quadrature_degree))
        dS_vp = dS_v(degree=(self.quadrature_degree))

        # add effect of density of water upon theta
        if self.moisture is not None:
            water_t = Function(Vtheta).assign(0.0)
            for water in self.moisture:
                water_t += self.state.fields(water)
            theta_w = theta / (1 + water_t)
            thetabar_w = thetabar / (1 + water_t)
        else:
            theta_w = theta
            thetabar_w = thetabar

        eqn = (
            inner(w, (state.h_project(u) - u_in))*dx
            - beta_cp*div(theta_w*V(w))*pibar*dxp
            # following does nothing but is preserved in the comments
            # to remind us why (because V(w) is purely vertical).
            # + beta_cp*jump(theta*V(w), n)*avg(pibar)*dS_v
            - beta_cp*div(thetabar_w*w)*pi*dxp
            + beta_cp*jump(thetabar_w*w, n)*avg(pi)*dS_vp
            + (phi*(rho - rho_in) - beta*inner(grad(phi), u)*rhobar)*dx
            + beta*jump(phi*u, n)*avg(rhobar)*(dS_v + dS_h)
        )

        if mu is not None:
            eqn += dt*mu*inner(w, k)*inner(u, k)*dx
        aeqn = lhs(eqn)
        Leqn = rhs(eqn)

        # Place to put result of u rho solver
        self.urho = Function(M)

        # Boundary conditions (assumes extruded mesh)
        bcs = [DirichletBC(M.sub(0), 0.0, "bottom"),
               DirichletBC(M.sub(0), 0.0, "top")]

        # Solver for u, rho
        urho_problem = LinearVariationalProblem(
            aeqn, Leqn, self.urho, bcs=bcs)

        self.urho_solver = LinearVariationalSolver(urho_problem,
                                                   solver_parameters=self.solver_parameters,
                                                   options_prefix='ImplicitSolver')

        # Reconstruction of theta
        theta = TrialFunction(Vtheta)
        gamma = TestFunction(Vtheta)

        u, rho = self.urho.split()
        self.theta = Function(Vtheta)

        theta_eqn = gamma*(theta - theta_in
                           + dot(k, u)*dot(k, grad(thetabar))*beta)*dx

        theta_problem = LinearVariationalProblem(lhs(theta_eqn),
                                                 rhs(theta_eqn),
                                                 self.theta)
        self.theta_solver = LinearVariationalSolver(theta_problem,
                                                    options_prefix='thetabacksubstitution')

    @timed_function("Gusto:SchurCompLinearSolve")
    def solve(self):
        """
        Apply the solver with rhs state.xrhs and result state.dy.
        """

        with timed_region("Gusto:VelocityDensitySolve"):
            self.urho_solver.solve()

        u1, rho1 = self.urho.split()
        u, rho, theta = self.state.dy.split()
        u.assign(u1)
        rho.assign(rho1)

        with timed_region("Gusto:ThetaRecon"):
            self.theta_solver.solve()

        theta.assign(self.theta)
