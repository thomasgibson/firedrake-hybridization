from firedrake import *


class GravityWaveSolver(object):
    """
    """

    def __init__(self, W2, W3, Wb, dt, c, N, Omega, R,
                 solver_type="AMG", hybridization=True, monitor=False):
        """
        """

        self.hybridization = hybridization
        self.monitor = monitor
        if solver_type == "AMG":
            self.up_params = self.amg_paramters
        elif solver_type == "GMG":
            self.up_params = self.gmg_parameters
        else:
            raise ValueError("Unknown inner solver type")

        self._dt = dt
        self._c = c
        self._N = N
        self._dt_half = Constant(0.5*dt)
        self._dt_half_N2 = Constant(0.5*dt*N**2)
        self._dt_half_c2 = Constant(0.5*dt*c**2)
        self._omega_N2 = Constant((0.5*dt*N)**2)

        self._Wmixed = W2 * W3
        self._W2 = self._Wmixed.sub(0)
        self._W3 = self._Wmixed.sub(1)
        self._Wb = Wb

        self._up = Function(self._Wmixed)
        self._b = Function(self._Wb)
        self._btmp = Function(self._Wb)

        self._state = Function(self._W2 * self._W3 * self._Wb, name="State")

        mesh = self._W3.mesh()
        # Not sure why this doesn't work...
        # self._khat = CellNormal(mesh)
        x = SpatialCoordinate(mesh)
        R = sqrt(inner(x, x))
        self._khat = interpolate(x/R, mesh.coordinates.function_space())

        fexpr = 2*Omega*x[2]/R
        Vcg = FunctionSpace(mesh, "CG", 1)
        self._f = interpolate(fexpr, Vcg)

        self._build_up_solver()
        self._build_b_solver()

    @property
    def amg_paramters(self):
        """
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
        """
        """

        u0, p0, b0 = self._state.split()

        utest, ptest = TestFunctions(self._Wmixed)
        utrial, ptrial = TrialFunctions(self._Wmixed)
        bcs = [DirichletBC(self._Wmixed.sub(0), 0.0, "bottom"),
               DirichletBC(self._Wmixed.sub(0), 0.0, "top")]

        a_up = (ptest*ptrial
                + self._dt_half_c2*ptest*div(utrial)
                - self._dt_half*div(utest)*ptrial
                + (dot(utest, utrial)
                   + self._dt_half*dot(utest, self._f*cross(self._khat, utrial))
                   + self._omega_N2
                    * dot(utest, self._khat)
                    * dot(utrial, self._khat))) * dx

        L_up = (dot(utest, u0)
                + self._dt_half*dot(utest, self._khat*b0)
                + ptest*p0) * dx

        up_problem = LinearVariationalProblem(a_up, L_up, self._up, bcs=bcs)
        up_solver = LinearVariationalSolver(up_problem,
                                            solver_parameters=self.up_params)
        self.up_solver = up_solver

    def _build_b_solver(self):
        """
        """

        u0, _, _ = self._state.split()

        btest = TestFunction(self._Wb)
        L_b = dot(btest*self._khat, u0) * dx
        a_b = btest*TrialFunction(self._Wb) * dx
        b_problem = LinearVariationalProblem(a_b, L_b, self._btmp)

        b_params = {'ksp_type': 'cg',
                    'pc_type': 'bjacobi',
                    'sub_pc_type': 'ilu'}
        if self.monitor:
            b_params['ksp_monitor_true_residual'] = True

        b_solver = LinearVariationalSolver(b_problem,
                                           solver_parameters=b_params)
        self.b_solver = b_solver

    def initialize(self, u, p, b):
        """
        """
        u0, p0, b0 = self._state.split()
        u0.assign(u)
        p0.assign(p)
        b0.assign(b)

    def solve(self):
        """
        """
        un, pn, bn = self._state.split()

        # Solve for u and p
        self._up.assign(0.0)
        self._b.assign(0.0)
        self.up_solver.solve()

        un.assign(self._up.sub(0))
        pn.assign(self._up.sub(1))

        # Reconstruct b
        self._btmp.assign(0.0)
        self.b_solver.solve()
        bn.assign(assemble(bn - self._dt_half_N2*self._btmp))
