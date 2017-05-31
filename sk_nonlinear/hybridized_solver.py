from __future__ import absolute_import
from firedrake import (split, LinearVariationalProblem,
                       LinearVariationalSolver, TestFunctions,
                       TrialFunctions, TestFunction, TrialFunction,
                       lhs, rhs, FacetNormal, div, dx, avg,
                       dS_v, dS_h, inner, MixedFunctionSpace, dot, grad,
                       Function, warning, FunctionSpace, BrokenElement,
                       ds_v, ds_t, ds_b, Tensor, assemble,
                       LinearSolver, Projector)
from gusto.forcing import exner, exner_rho, exner_theta
from gusto.linear_solver import TimesteppingSolver


class HybridizedCompressibleSolver(TimesteppingSolver):
    """
    Timestepping linear solver object for the compressible equations
    in theta-pi formulation with prognostic variables u,rho,theta.

    This solver follows the following strategy:
    (1) Analytically eliminate theta (introduces error near topography)
    (2a) Solve resulting system for (u[broken],rho,lambda) using hybridised
    solver
    (2b) reconstruct unbroken u
    (3) Reconstruct theta

    :arg state: a :class:`.State` object containing everything else.
    :arg quadrature degree: tuple (q_h, q_v) where q_h is the required
    quadrature degree in the horizontal direction and q_v is that in
    the vertical direction
    :arg params (optional): solver parameters
    """

    def __init__(self, state, quadrature_degree=None, params=None):

        self.state = state

        if quadrature_degree is not None:
            self.quadrature_degree = quadrature_degree
        else:
            dgspace = state.spaces("DG")
            if any(deg > 2 for deg in dgspace.ufl_element().degree()):
                warning(
                    "Default quadrature degree most likely not sufficient "
                    "for this degree element."
                )
            self.quadrature_degree = (5, 5)

        self.params = params

        # setup the solver
        self._setup_solver()
        print("Oooh, he's trying")

    def _setup_solver(self):
        state = self.state      # just cutting down line length a bit
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

        # Build the reduced function space for u,rho
        M = MixedFunctionSpace((Vu_broken, Vrho))
        w, phi = TestFunctions(M)
        u, rho = TrialFunctions(M)

        l0 = TrialFunction(Vtrace)
        dl = TestFunction(Vtrace)

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

        # the pi prime term (here, bars are for mean and no bars are
        # for linear perturbations)

        pi = pibar_theta*theta + pibar_rho*rho

        # vertical projection
        def V(u):
            return k*inner(u, k)

        # specify degree for some terms as estimated degree is too large
        dxp = dx(degree=(self.quadrature_degree))
        dS_vp = dS_v(degree=(self.quadrature_degree))
        dS_hp = dS_h(degree=(self.quadrature_degree))
        ds_vp = ds_v(degree=(self.quadrature_degree))
        ds_tbp = ds_t(degree=(self.quadrature_degree)) + ds_b(degree=(self.quadrature_degree))

        rhobar_tr = Function(Vtrace)
        rbareqn = (l0('+') - avg(rhobar))*dl('+')*(dS_vp + dS_hp) + \
                  (l0 - rhobar)*dl*ds_vp + \
                  (l0 - rhobar)*dl*ds_tbp
        rhobar_prob = LinearVariationalProblem(lhs(rbareqn), rhs(rbareqn), rhobar_tr)
        self.rhobar_solver = LinearVariationalSolver(rhobar_prob,
                                                     solver_parameters={'ksp_type': 'preonly',
                                                                        'pc_type': 'bjacobi',
                                                                        'pc_sub_type': 'lu'})

        Aeqn = (
            inner(w, (u - u_in))*dx
            - beta*cp*div(theta*V(w))*pibar*dxp
            - beta*cp*div(thetabar*w)*pi*dxp
            + (phi*(rho - rho_in) - beta*inner(grad(phi), u)*rhobar)*dx
            + beta*inner(phi*u, n)*rhobar_tr*(dS_v + dS_h)
        )
        if mu is not None:
            Aeqn += dt*mu*inner(w, k)*inner(u, k)*dx
        Aop = Tensor(lhs(Aeqn))
        Arhs = rhs(Aeqn)

        #  (A K)(U) = (U_r)
        #  (L 0)(l)   (0  )

        dl = dl('+')
        l0 = l0('+')

        K = Tensor(beta*cp*inner(thetabar*w, n)*l0*(dS_vp + dS_hp)
                   + beta*cp*inner(thetabar*w, n)*l0*ds_vp
                   + beta*cp*inner(thetabar*w, n)*l0*ds_tbp)
        L = Tensor(dl*inner(u, n)*(dS_vp + dS_hp)
                   + dl*inner(u, n)*ds_vp
                   + dl*inner(u, n)*ds_tbp)

        #  U = A^{-1}(-Kl + U_r), 0=LU=-(LA^{-1}K)l + LA^{-1}U_r, so (LA^{-1}K)l = LA^{-1}U_r
        # reduced eqns for l0
        S = assemble(L * Aop.inv * K)
        self.Rexp = L * Aop.inv * Tensor(Arhs)
        # place to put the RHS (self.Rexp gets assembled in here)
        self.R = Function(M)

        # Set up the LinearSolver for the system of Lagrange multipliers
        self.lSolver = LinearSolver(S, solver_parameters=self.params)
        # a place to keep the solution
        self.lambdar = Function(Vtrace)

        # Place to put result of u rho solver
        self.urho = Function(M)

        # Reconstruction of broken u and rho
        self.ASolver = LinearSolver(assemble(Aop),
                                    solver_parameters={'ksp_type': 'preonly',
                                                       'pc_type': 'bjacobi',
                                                       'pc_sub_type': 'lu'},
                                    options_prefix='urhoreconstruction')
        # Rhs for broken u and rho reconstruction
        self.Rurhoexp = -K*self.lambdar + Tensor(Arhs)
        self.Rurho = Function(M)
        u, rho = self.urho.split()

        self.u_hdiv = Function(Vu)
        self.u_projector = Projector(u, self.u_hdiv,
                                     solver_parameters={'ksp_type': 'cg',
                                                        'pc_type': 'bjacobi',
                                                        'pc_sub_type': 'ilu'})

        # Reconstruction of theta
        theta = TrialFunction(Vtheta)
        gamma = TestFunction(Vtheta)

        self.theta = Function(Vtheta)

        u = self.u_hdiv
        theta_eqn = gamma*(theta - theta_in +
                           dot(k, u)*dot(k, grad(thetabar))*beta)*dx

        theta_problem = LinearVariationalProblem(lhs(theta_eqn),
                                                 rhs(theta_eqn),
                                                 self.theta)
        self.theta_solver = LinearVariationalSolver(theta_problem,
                                                    solver_parameters={'ksp_type': 'cg',
                                                                       'pc_type': 'bjacobi',
                                                                       'pc_sub_type': 'ilu'},
                                                    options_prefix='thetabacksubstitution')

    def solve(self):
        """
        Apply the solver with rhs state.xrhs and result state.dy.
        """

        # Assemble the RHS for lambda into self.R
        assemble(self.Rexp, tensor=self.R)
        # Solve for lambda
        self.lSolver.solve(self.lambdar, self.R)
        # Assemble the RHS for uhat, rho reconstruction
        assemble(self.Rurhoexp, tensor=self.Rurho)
        # Solve for uhat, rho
        self.ASolver.solve(self.urho, self.Rurho)
        # Project uhat as self.u_hdiv in H(div)
        self.u_projector.project()
        # copy back into u and rho cpts of dy
        _, rho1 = self.urho.split()
        u, rho, theta = self.state.dy.split()
        u.assign(self.u_hdiv)
        rho.assign(rho1)
        # reconstruct theta
        self.theta_solver.solve()
        # copy into theta cpt of dy
        theta.assign(self.theta)
