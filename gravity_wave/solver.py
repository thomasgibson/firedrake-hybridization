from firedrake import *
from pyop2.profiling import timed_region, timed_function


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

    @timed_function("SolverInit")
    def __init__(self, W2, W3, Wb, dt, c, N, Omega, R,
                 solver_type="AMG", hybridization=True, monitor=False):
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
        :solver_type: A string describing which inner-most solver to use on
                      the pressure space (approximate Schur-complement) or
                      the trace space (hybridization). Currently, only the
                      parameter "AMG" is supported, which uses smoothed
                      aggregation algebraic multigrid (GAMG).
        :arg hybridization: A boolean switch between using a hybridized
                            mixed method (True) on the velocity-pressure
                            system, or GMRES with an approximate Schur-
                            complement preconditioner (False). The default
                            is set to `True`
                            (because hybridization is awesome).
        :arg monitor: A boolean switch with turns on/off KSP monitoring
                      of the problem residuals (primarily for debugging
                      and checking convergence of the solver). When profiling,
                      keep this set to `False`.
        """

        self.hybridization = hybridization
        self.monitor = monitor
        if solver_type == "AMG":
            self.up_params = self.amg_paramters
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

        # Functions for state solutions
        self._up = Function(self._Wmixed)
        self._b = Function(self._Wb)
        self._btmp = Function(self._Wb)

        self._state = Function(self._W2 * self._W3 * self._Wb, name="State")

        # Outward normal vector
        mesh = self._W3.mesh()
        x = SpatialCoordinate(mesh)
        R = sqrt(inner(x, x))
        self._khat = interpolate(x/R, mesh.coordinates.function_space())

        # Coriolis term
        fexpr = 2*Omega*x[2]/R
        Vcg = FunctionSpace(mesh, "CG", 1)
        self._f = interpolate(fexpr, Vcg)

        # Construct linear solvers
        with timed_region("SolverSetup"):
            self._build_up_solver()
            self._build_b_solver()

    @property
    def amg_paramters(self):
        """Solver parameters for the velocity-pressure system using
        algebraic multigrid. Current tolerance is set to 5 digits of
        accuracy.
        """

        if self.hybridization:
            params = {'ksp_type': 'preonly',
                      'mat_type': 'matfree',
                      'pc_type': 'python',
                      'pc_python_type': 'firedrake.HybridizationPC',
                      'hybridization': {'ksp_type': 'cg',
                                        'pc_type': 'gamg',
                                        'ksp_rtol': 1e-5,
                                        'mg_levels': {'ksp_type': 'chebyshev',
                                                      'ksp_max_it': 1,
                                                      'pc_type': 'bjacobi',
                                                      'sub_pc_type': 'ilu'}}}
            if self.monitor:
                params['hybridization']['ksp_monitor_true_residual'] = True
        else:
            params = {'ksp_type': 'gmres',
                      'ksp_rtol': 1e-5,
                      'pc_type': 'fieldsplit',
                      'pc_fieldsplit_type': 'schur',
                      'ksp_type': 'gmres',
                      'ksp_max_it': 100,
                      'ksp_gmres_restart': 50,
                      'pc_fieldsplit_schur_fact_type': 'FULL',
                      'pc_fieldsplit_schur_precondition': 'selfp',
                      'fieldsplit_0': {'ksp_type': 'preonly',
                                       'pc_type': 'bjacobi',
                                       'sub_pc_type': 'ilu'},
                      'fieldsplit_1': {'ksp_type': 'preonly',
                                       'pc_type': 'gamg',
                                       'mg_levels': {'ksp_type': 'chebyshev',
                                                     'ksp_max_it': 1,
                                                     'pc_type': 'bjacobi',
                                                     'sub_pc_type': 'ilu'}}}
            if self.monitor:
                params['ksp_monitor_true_residual'] = True
        return params

    def _build_up_solver(self):
        """Constructs the solver for the velocity-pressure increments."""

        # Current state
        u0, p0, b0 = self._state.split()

        # Test and trial functions for the velocity-pressure increment
        # system (with strong no-slip boundary conditions on the top
        # and bottom of the atmospheric domain)
        utest, ptest = TestFunctions(self._Wmixed)
        utrial, ptrial = TrialFunctions(self._Wmixed)
        bcs = [DirichletBC(self._Wmixed.sub(0), 0.0, "bottom"),
               DirichletBC(self._Wmixed.sub(0), 0.0, "top")]

        def outward(u):
            return cross(self._khat, u)

        # Linear gravity wave system for the velocity and pressure
        # increments (buoyancy has been eliminated in the discrete
        # equations since there is no orography)
        a_up = (ptest*ptrial
                + self._dt_half_c2*ptest*div(utrial)
                - self._dt_half*div(utest)*ptrial
                + (dot(utest, utrial)
                   + self._dt_half*dot(utest, self._f*outward(utrial))
                   + self._omega_N2
                    * dot(utest, self._khat)
                    * dot(utrial, self._khat))) * dx

        L_up = (dot(utest, u0)
                + self._dt_half*dot(utest, self._khat*b0)
                + ptest*p0) * dx

        # Set up linear solver
        up_problem = LinearVariationalProblem(a_up, L_up, self._up, bcs=bcs)
        up_solver = LinearVariationalSolver(up_problem,
                                            solver_parameters=self.up_params)
        self.up_solver = up_solver

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

    @timed_function("FullSolve")
    def solve(self):
        """Solves the linear gravity wave problem at a particular
        time-step in two-stages. First, the velocity and pressure
        solutions are computed, then buoyancy is reconstructed from
        the computed fields. The solver state is then updated.
        """

        # Previous states
        un, pn, bn = self._state.split()

        # Solve for u and p updates
        self._up.assign(0.0)
        self._b.assign(0.0)
        with timed_region("Velocity-Pressure-Solve"):
            self.up_solver.solve()

        un.assign(self._up.sub(0))
        pn.assign(self._up.sub(1))

        # Reconstruct b
        self._btmp.assign(0.0)
        with timed_region("Buoyancy-Solve"):
            self.b_solver.solve()

        bn.assign(assemble(bn - self._dt_half_N2*self._btmp))
