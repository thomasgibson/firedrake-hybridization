from firedrake import *
from firedrake.assemble import create_assembly_callable
from firedrake.parloops import par_loop, READ, INC
from firedrake.utils import cached_property
from pyop2.profiling import timed_stage, timed_region
from ksp_monitor import KSPMonitorDummy
import numpy as np
import time


class GravityWaveSolver(object):
    """Solver for the linearized compressible Boussinesq equations
    (includes Coriolis term). The equations are solved in three stages:

    (1) First analytically eliminate the buoyancy perturbation term from
        the discrete equations. This is possible since there is currently
        no orography. Note that it is indeed possible to eliminate buoyancy
        when orography is present, however this must be done at the continuous
        level first.

    (2) Eliminating buoyancy produces a saddle-point system for the velocity
        and pressure perturbations. The resulting system is solved using
        either an approximate full Schur-complement procedure or a
        hybridized mixed method.

    (3) Once the velocity and perturbation fields are computed from the
        previous step, the buoyancy term is reconstructed.
    """

    def __init__(self, W2, W3, Wb, dt, c, N, Omega, R, rtol=1.0E-6,
                 solver_type="gamg", hybridization=False,
                 local_invert_method=None,
                 local_solve_method=None,
                 monitor=False):
        """The constructor for the GravityWaveSolver.

        :arg W2: The HDiv velocity space.
        :arg W3: The L2 pressure space.
        :arg Wb: The "Charney-Phillips" space for the buoyancy field.
        :arg dt: A positive real number denoting the time-step size.
        :arg c: A positive real number denoting the speed of sound waves
                in dry air.
        :arg N: A positive real number describing the Brunt–Väisälä frequency.
        :arg Omega: A positive real number; the angular rotation rate of the
                    Earth.
        :arg R: A positive real number denoting the radius of the spherical
                mesh (Earth-size).
        :arg rtol: The relative tolerance for the solver.
        :solver_type: A string describing which inner-most solver to use on
                      the pressure space (approximate Schur-complement) or
                      the trace space (hybridization). Currently, only the
                      parameter "AMG" is supported, which uses smoothed
                      aggregation algebraic multigrid (GAMG).
        :arg hybridization: A boolean switch between using a hybridized
                            mixed method (True) on the velocity-pressure
                            system, or GMRES with an approximate Schur-
                            complement preconditioner (False).
        :arg local_invert_method: Optional argument detailing what kind of
                                  factorization to perform in Eigen when
                                  computing local inverses.
        :arg local_solve_method: Optional argument detailing what kind of
                                 factorization to perform in Eigen when
                                 computing the local solves in the hybridized
                                 solver.
        :arg monitor: A boolean switch with turns on/off KSP monitoring
                      of the problem residuals (primarily for debugging
                      and checking convergence of the solver). When profiling,
                      keep this set to `False`.
        """

        self.hybridization = hybridization
        self._local_solve_method = local_solve_method
        self._local_invert_method = local_invert_method
        self.monitor = monitor
        self.rtol = rtol
        if solver_type == "gamg":
            self.params = self.gamg_paramters
        elif solver_type == "hypre":
            self.params = self.hypre_parameters
        elif solver_type == "direct":
            self.params = self.direct_parameters
        else:
            raise ValueError("Unknown inner solver type")

        # Timestepping parameters and physical constants
        self._dt = dt
        self._c = c
        self._N = N
        self._dt_half = Constant(0.5*dt)
        self._dt_half_N2 = Constant(0.5*dt*N**2)
        self._dt_half_c2 = Constant(0.5*dt*c**2)
        self._omega_N2 = Constant((0.5*dt*N)**2)

        # Compatible finite element spaces
        self._Wmixed = W2 * W3
        self._W2 = self._Wmixed.sub(0)
        self._W3 = self._Wmixed.sub(1)
        self._Wb = Wb

        mesh = self._W3.mesh()

        # Hybridized finite element spaces
        broken_W2 = BrokenElement(self._W2.ufl_element())
        self._W2disc = FunctionSpace(mesh, broken_W2)

        h_deg, v_deg = self._W2.ufl_element().degree()
        tdegree = (h_deg - 1, v_deg - 1)
        self._WT = FunctionSpace(mesh, "HDiv Trace", tdegree)
        self._Whybrid = self._W2disc * self._W3 * self._WT
        self._hybrid_update = Function(self._Whybrid)
        self._facet_normal = FacetNormal(mesh)

        shapes = (self._W2.finat_element.space_dimension(),
                  np.prod(self._W2.shape))
        weight_kernel = """
        for (int i=0; i<%d; ++i) {
        for (int j=0; j<%d; ++j) {
        w[i][j] += 1.0;
        }}""" % shapes

        self.weight = Function(self._W2)
        par_loop(weight_kernel, dx, {"w": (self.weight, INC)})
        self.average_kernel = """
        for (int i=0; i<%d; ++i) {
        for (int j=0; j<%d; ++j) {
        vec_out[i][j] += vec_in[i][j]/w[i][j];
        }}""" % shapes

        # Functions for state solutions
        self._up = Function(self._Wmixed)
        self._b = Function(self._Wb)
        self._btmp = Function(self._Wb)

        self._state = Function(self._W2 * self._W3 * self._Wb, name="State")

        # Outward normal vector
        x = SpatialCoordinate(mesh)
        R = sqrt(inner(x, x))
        self._khat = interpolate(x/R, mesh.coordinates.function_space())

        # Coriolis term
        fexpr = 2*Omega*x[2]/R
        Vcg = FunctionSpace(mesh, "CG", 1)
        self._f = interpolate(fexpr, Vcg)

        # Construct linear solvers
        if self.hybridization:
            self._build_hybridized_solver()
        else:
            self._build_up_solver()

        self._build_b_solver()

        self._ksp_monitor = KSPMonitorDummy()
        self.up_residual_reductions = []

    @property
    def direct_parameters(self):
        """Solver parameters using a direct method (LU)"""

        inner_params = {'ksp_type': 'preonly',
                        'pc_type': 'lu',
                        'pc_factor_mat_solver_package': 'mumps'}

        if self.hybridization:
            params = inner_params
        else:
            params = {'ksp_type': 'preonly',
                      'pc_type': 'fieldsplit',
                      'pc_fieldsplit_type': 'schur',
                      'pc_fieldsplit_schur_fact_type': 'FULL',
                      'fieldsplit_0': inner_params,
                      'fieldsplit_1': inner_params}

        return params

    @property
    def hypre_parameters(self):
        """Solver parameters using hypre's boomeramg
        implementation of AMG.
        """

        inner_params = {'ksp_type': 'cg',
                        'ksp_rtol': self.rtol,
                        'pc_type': 'hypre',
                        'pc_hypre_type': 'boomeramg',
                        'pc_hypre_boomeramg_no_CF': False,
                        'pc_hypre_boomeramg_coarsen_type': 'HMIS',
                        'pc_hypre_boomeramg_interp_type': 'ext+i',
                        'pc_hypre_boomeramg_P_max': 0,
                        'pc_hypre_boomeramg_agg_nl': 0,
                        'pc_hypre_boomeramg_max_level': 5,
                        'pc_hypre_boomeramg_strong_threshold': 0.25}

        if self.monitor:
            inner_params['ksp_monitor_true_residual'] = True

        if self.hybridization:
            params = inner_params
        else:
            params = {'ksp_type': 'gmres',
                      'ksp_rtol': self.rtol,
                      'pc_type': 'fieldsplit',
                      'pc_fieldsplit_type': 'schur',
                      'ksp_max_it': 100,
                      'ksp_gmres_restart': 50,
                      'pc_fieldsplit_schur_fact_type': 'FULL',
                      'pc_fieldsplit_schur_precondition': 'selfp',
                      'fieldsplit_0': {'ksp_type': 'preonly',
                                       'pc_type': 'bjacobi',
                                       'sub_pc_type': 'ilu'},
                      'fieldsplit_1': inner_params}
            if self.monitor:
                params['ksp_monitor_true_residual'] = True

        return params

    @property
    def gamg_paramters(self):
        """Solver parameters for the velocity-pressure system using
        algebraic multigrid.
        """

        inner_params = {'ksp_type': 'cg',
                        'pc_type': 'gamg',
                        'ksp_rtol': self.rtol,
                        'mg_levels': {'ksp_type': 'chebyshev',
                                      'ksp_max_it': 2,
                                      'pc_type': 'bjacobi',
                                      'sub_pc_type': 'ilu'}}
        if self.monitor:
            inner_params['ksp_monitor_true_residual'] = True

        if self.hybridization:
            params = inner_params
        else:
            params = {'ksp_type': 'gmres',
                      'ksp_rtol': self.rtol,
                      'pc_type': 'fieldsplit',
                      'pc_fieldsplit_type': 'schur',
                      'ksp_max_it': 100,
                      'ksp_gmres_restart': 50,
                      'pc_fieldsplit_schur_fact_type': 'FULL',
                      'pc_fieldsplit_schur_precondition': 'selfp',
                      'fieldsplit_0': {'ksp_type': 'preonly',
                                       'pc_type': 'bjacobi',
                                       'sub_pc_type': 'ilu'},
                      'fieldsplit_1': inner_params}
            if self.monitor:
                params['ksp_monitor_true_residual'] = True

        return params

    @cached_property
    def _build_up_bilinear_form(self):
        """Bilinear form for the gravity wave velocity-pressure
        subsystem.
        """

        utest, ptest = TestFunctions(self._Wmixed)
        u, p = TrialFunctions(self._Wmixed)

        def outward(u):
            return cross(self._khat, u)

        # Linear gravity wave system for the velocity and pressure
        # increments (buoyancy has been eliminated in the discrete
        # equations since there is no orography)
        a_up = (ptest*p
                + self._dt_half_c2*ptest*div(u)
                - self._dt_half*div(utest)*p
                + (dot(utest, u)
                   + self._dt_half*dot(utest, self._f*outward(u))
                   + self._omega_N2
                   * dot(utest, self._khat)
                   * dot(u, self._khat))) * dx
        return a_up

    @cached_property
    def _build_hybridized_bilinear_form(self):
        """Bilinear form for the hybrid-mixed velocity-pressure
        subsystem.
        """

        utest, ptest, lambdatest = TestFunctions(self._Whybrid)

        u, p, lambdar = TrialFunctions(self._Whybrid)

        def outward(u):
            return cross(self._khat, u)

        n = self._facet_normal

        # Hybridized linear gravity wave system for the velocity,
        # pressure, and trace subsystem (buoyancy has been eliminated
        # in the discrete equations since there is no orography).
        # NOTE: The no-slip boundary conditions are applied weakly
        # in the hybridized problem.
        a_uplambdar = ((ptest*p
                        + self._dt_half_c2*ptest*div(u)
                        - self._dt_half*div(utest)*p
                        + (dot(utest, u)
                           + self._dt_half*dot(utest, self._f*outward(u))
                           + self._omega_N2
                           * dot(utest, self._khat)
                           * dot(u, self._khat))) * dx
                       + lambdar * jump(utest, n=n) * (dS_v + dS_h)
                       + lambdar * dot(utest, n) * ds_tb
                       + lambdatest * jump(u, n=n) * (dS_v + dS_h)
                       + lambdatest * dot(u, n) * ds_tb)

        return a_uplambdar

    def _build_up_rhs(self, u0, p0, b0):
        """Right-hand side for the gravity wave velocity-pressure
        subsystem.
        """

        def outward(u):
            return cross(self._khat, u)

        utest, ptest = TestFunctions(self._Wmixed)
        L_up = (dot(utest, u0)
                + self._dt_half*dot(utest, self._f*outward(u0))
                + self._dt_half*dot(utest, self._khat*b0)
                + ptest*p0) * dx

        return L_up

    def _build_hybridized_rhs(self, u0, p0, b0):
        """Right-hand side for the hybridized gravity wave
        velocity-pressure-trace subsystem.
        """

        def outward(u):
            return cross(self._khat, u)

        # No residual for the traces; they only enforce continuity
        # of the discontinuous velocity normals
        utest, ptest, _ = TestFunctions(self._Whybrid)
        L_uplambdar = (dot(utest, u0)
                       + self._dt_half*dot(utest, self._f*outward(u0))
                       + self._dt_half*dot(utest, self._khat*b0)
                       + ptest*p0) * dx

        return L_uplambdar

    def up_residual(self, old_state, new_up):
        """Returns the residual of the velocity-pressure system."""

        u0, p0, b0 = old_state.split()
        res = self._build_up_rhs(u0, p0, b0)
        L = self._build_up_bilinear_form
        res -= action(L, new_up)

        return res

    def _build_up_solver(self):
        """Constructs the solver for the velocity-pressure increments."""

        # strong no-slip boundary conditions on the top
        # and bottom of the atmospheric domain)
        bcs = [DirichletBC(self._Wmixed.sub(0), 0.0, "bottom"),
               DirichletBC(self._Wmixed.sub(0), 0.0, "top")]

        # Mixed operator
        A = assemble(self._build_up_bilinear_form, bcs=bcs)

        # Set up linear solver
        linear_solver = LinearSolver(A, solver_parameters=self.params)
        self.linear_solver = linear_solver

        # Function to store RHS for the linear solver
        u0, p0, b0 = self._state.split()
        self._up_rhs = Function(self._Wmixed)
        self._assemble_up_rhs = create_assembly_callable(
            self._build_up_rhs(u0, p0, b0),
            tensor=self._up_rhs)

    def _build_hybridized_solver(self):
        """Constructs the Schur-complement system for the hybridized
        problem. In addition, all reconstruction calls are generated
        for recovering velocity and pressure.
        """

        # Matrix operator has the form:
        #  | A00 A01 A02 |
        #  | A10 A11  0  |
        #  | A20  0   0  |
        # for the U-Phi-Lambda system.
        # Create Slate tensors for the 3x3 block operator:
        A = Tensor(self._build_hybridized_bilinear_form)

        # Define the 2x2 mixed block:
        #  | A00 A01 |
        #  | A10 A11 |
        # which couples the potential and momentum.
        Atilde = A.block(((0, 1), (0, 1)))

        # and the off-diagonal blocks:
        # |A20 0| & |A02 0|^T:
        Q = A.block((2, (0, 1)))
        Qt = A.block(((0, 1), 2))

        # Schur complement operator:
        S = assemble(Q * Atilde.inv(self._local_invert_method) * Qt)

        # Set up linear solver
        linear_solver = LinearSolver(S, solver_parameters=self.params)
        self.linear_solver = linear_solver

        # Tensor for the residual
        u0, p0, b0 = self._state.split()
        R = Tensor(self._build_hybridized_rhs(u0, p0, b0))
        R01 = R.block(((0, 1),))

        # Function to store the rhs for the trace system
        self._S_rhs = Function(self._WT)
        self._assemble_Srhs = create_assembly_callable(
            Q * Atilde.inv(self._local_invert_method) * R01,
            tensor=self._S_rhs)

        # Individual blocks: 0 indices correspond to u coupling;
        # 1 corresponds to p coupling; and 2 is trace coupling.
        A00 = A.block((0, 0))
        A01 = A.block((0, 1))
        A10 = A.block((1, 0))
        A11 = A.block((1, 1))
        A02 = A.block((0, 2))
        R0 = R.block((0,))
        R1 = R.block((1,))

        # Local coefficient vectors
        Lambda = AssembledVector(self._hybrid_update.sub(2))
        P = AssembledVector(self._hybrid_update.sub(1))

        Sp = A11 - A10 * A00.inv(self._local_invert_method) * A01
        p_problem = Sp.solve(R1 - A10 *
                             A00.inv(self._local_invert_method) *
                             (R0 - A02 * Lambda),
                             method=self._local_solve_method)

        u_problem = A00.solve(R0 - A01 * P - A02 * Lambda,
                              method=self._local_solve_method)

        # Two-stage reconstruction
        self._assemble_pressure = create_assembly_callable(
            p_problem, tensor=self._hybrid_update.sub(1))
        self._assemble_velocity = create_assembly_callable(
            u_problem, tensor=self._hybrid_update.sub(0))

    @property
    def ksp_monitor(self):
        """Returns the KSP monitor attached to this solver. Note
        that the monitor is for the velocity-pressure system.
        """

        return self._ksp_monitor

    @ksp_monitor.setter
    def ksp_monitor(self, kspmonitor):
        """Set the monitor for the velocity-pressure or trace system.

        :arg kspmonitor: a monitor to use.
        """

        self._ksp_monitor = kspmonitor
        ksp = self.linear_solver.ksp
        ksp.setMonitor(self._ksp_monitor)

    def _build_b_solver(self):
        """Constructs the solver for the buoyancy update."""

        # Computed velocity perturbation
        u0, _, _ = self._state.split()

        # Expression for buoyancy reconstruction
        btest = TestFunction(self._Wb)
        L_b = dot(btest*self._khat, u0) * dx
        a_b = btest*TrialFunction(self._Wb) * dx
        b_problem = LinearVariationalProblem(a_b, L_b, self._btmp)

        b_params = {'ksp_type': 'cg',
                    'pc_type': 'bjacobi',
                    'sub_pc_type': 'ilu'}
        if self.monitor:
            b_params['ksp_monitor_true_residual'] = True

        # Solver for buoyancy update
        b_solver = LinearVariationalSolver(b_problem,
                                           solver_parameters=b_params)
        self.b_solver = b_solver

    def initialize(self, u, p, b):
        """Initialized the solver state with initial conditions
        for the velocity, pressure, and buoyancy fields.

        :arg u: An initial condition (`firedrake.Function`)
                for the velocity field.
        :arg p: An initial condition for the pressure field.
        :arg b: And finally an function describing the initial
                state of the buoyancy field.
        """

        u0, p0, b0 = self._state.split()
        u0.assign(u)
        p0.assign(p)
        b0.assign(b)

    def solve(self):
        """Solves the linear gravity wave problem at a particular
        time-step in two-stages. First, the velocity and pressure
        solutions are computed, then buoyancy is reconstructed from
        the computed fields. The solver state is then updated.
        """

        # Previous state
        un, pn, bn = self._state.split()

        # Initial residual
        self._hybrid_update.assign(0.0)
        self._up.assign(0.0)
        self._b.assign(0.0)
        r0 = assemble(self.up_residual(self._state, self._up))

        # Main solver stage
        t_start = time.time()
        with timed_stage("Velocity-Pressure-Solve"):
            if self.hybridization:

                # Solve for the Lagrange multipliers
                with timed_region("Trace-Solver"):
                    self._assemble_Srhs()
                    self.linear_solver.solve(self._hybrid_update.sub(2),
                                             self._S_rhs)

                # Recover pressure, then velocity
                with timed_region("Hybrid-Reconstruct"):
                    self._assemble_pressure()
                    self._assemble_velocity()

                    # Transfer hybridized solutions to the conforming spaces
                    self._up.sub(1).assign(self._hybrid_update.sub(1))
                    par_loop(self.average_kernel, dx,
                             {"w": (self.weight, READ),
                              "vec_in": (self._hybrid_update.sub(0), READ),
                              "vec_out": (self._up.sub(0), INC)})
            else:
                self._assemble_up_rhs()
                self.linear_solver.solve(self._up, self._up_rhs)
        t_finish = time.time()
        print ('   elapsed time = ', t_finish-t_start)

        # Residual after solving
        rn = assemble(self.up_residual(self._state, self._up))
        self.up_residual_reductions.append(rn.dat.norm/r0.dat.norm)

        # Update state
        un.assign(self._up.sub(0))
        pn.assign(self._up.sub(1))

        # Reconstruct b
        self._btmp.assign(0.0)
        with timed_stage("Buoyancy-Solve"):
            self.b_solver.solve()
            bn.assign(assemble(bn - self._dt_half_N2*self._btmp))
