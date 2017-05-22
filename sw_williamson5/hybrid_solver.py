from __future__ import absolute_import, print_function, division

from firedrake import *

from gusto import TimesteppingSolver as Solver


class SWEHybridSolver(Solver):

    def _setup_solver(self):

        state = self.state
        H = state.parameters.H
        g = state.parameters.g
        beta = state.timestepping.dt*state.timestepping.alpha

        # Split rhs vector symbolically
        u_in, D_in = split(state.xrhs)

        self.W = state.W
        w, phi = TestFunctions(self.W)
        u, D = TrialFunctions(self.W)

        eqn = (inner(w, u) - beta*g*div(w)*D - inner(w, u_in)
               + phi*D + beta*H*phi*div(u) - phi*D_in)*dx
        self.a = lhs(eqn)
        self.L = rhs(eqn)

        # Place to put result of mixed solution
        self.uD = Function(self.W)

    def manual_hybridization(self):
        print("Manually hybridizing the system...")

        # Break the mixed space:
        print("Manually breaking the finite element spaces...")
        Wd = FunctionSpace(self.W.mesh(),
                           MixedElement([BrokenElement(V.ufl_element())
                                         for V in self.W]))
        # Place to put the broken result
        self.uD_broken = Function(Wd)

        # Replace the arguments of the bilinear form with their
        # broken counterparts
        print("Replacing with broken arguments in the bilinear form...")
        self.deqn = replace(self.a, dict(zip(self.a.arguments(),
                                             (TestFunction(Wd),
                                              TrialFunction(Wd)))))

        # Create Slate tensor for the broken operator
        print("Creating Slate tensor for the broken operator...")
        Atilde = Tensor(self.deqn)

        # Introduce Lagrange multipliers
        print("Introducing Lagrange multipliers and constructing tensors...")
        sigma, _ = TrialFunctions(Wd)
        T = FunctionSpace(self.W.mesh(), "HDiv Trace",
                          self.W.ufl_element().degree())

        # Function for multipliers
        self.lambdar = Function(T)

        gammar = TestFunction(T)
        self.trace_form = gammar('+')*inner(sigma,
                                            FacetNormal(self.W.mesh()))*dS
        K = Tensor(self.trace_form)

        trace_bcs = DirichletBC(T, Constant(0.0), "on_boundary")

        # Assemble schur complement
        print("Constructing the Schur system via local eliminations...")
        self.S = assemble(K * Atilde.inv * K.T, bcs=trace_bcs)
        L = assemble(self.L)
        f, g = L.split()
        self.L_b = Function(Wd)
        f_b, g_b = self.L_b.split()
        project(f, f_b)
        interpolate(g, g_b)
        self.R = assemble(K * Atilde.inv * self.L_b)

    def solve(self):
        from firedrake.formmanipulation import split_form

        # Initial setup
        self.manual_hybridization()

        # Solving globally for Lambda
        print("Solving global system for the Lagrange multipliers...")
        solve(self.S, self.lambdar, self.R,
              solver_parameters={"pc_type": "lu",
                                 "ksp_type": "preonly"})

        print("Performing local solves for reconstruction...")
        sigma, u = self.uD_broken.split()
        split_mixed_op = dict(split_form(self.deqn))
        split_trace_op = dict(split_form(self.trace_form))

        split_rhs = self.L_b.split()
        g = split_rhs[0]
        f = split_rhs[1]
        A = Tensor(split_mixed_op[(0, 0)])
        B = Tensor(split_mixed_op[(0, 1)])
        C = Tensor(split_mixed_op[(1, 0)])
        D = Tensor(split_mixed_op[(1, 1)])
        K_0 = Tensor(split_trace_op[(0, 0)])
        K_1 = Tensor(split_trace_op[(0, 1)])

        M = D - C * A.inv * B
        R = K_1.T - C * A.inv * K_0.T
        u_rec = M.inv * f - M.inv * (C * A.inv * g + R * self.lambdar)
        print("Assembling broken pressure...")
        assemble(u_rec, tensor=u)

        print("Assembling broken velocity...")
        sigma_rec = A.inv * g - A.inv * (B * u + K_0.T * self.lambdar)
        assemble(sigma_rec, tensor=sigma)

        print("Projecting broken solutions into the mimetic spaces...")
        # velocity, pressure = self.uD.split()
        # project(sigma, velocity)
        # interpolate(u, pressure)

        update_vel, update_pr = self.state.dy.split()
        project(sigma, update_vel)
        interpolate(u, update_pr)
