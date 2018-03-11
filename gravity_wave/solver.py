from firedrake import *


class GravityWaveSolver(object):
    """
    """
    params = {
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

    b_params = {
        'ksp_type': 'cg',
        'pc_type': 'bjacobi',
        'sub_pc_type': 'ilu'
    }

    def __init__(self, mesh, dt, W2, W3, Wb):
        """
        """
        self.alpha = 0.5
        self.dt = dt
        self.beta = self.alpha * self.dt
        self.mesh = mesh
        self.W = W2 * W3 * Wb
        self.N = 0.01      # Brunt-Vaisala frequency (1/s)
        self.c = 343.      # Speed of sound in dry air (m/s)

        kvec = [0.0, 1.0]
        self.k = Constant(kvec)

        self.xn = Function(self.W, name="State")
        self._Wup = W2 * W3
        self.up = Function(self._Wup)
        self._Wb = Wb
        self.b = Function(self._Wb)

        self._setup_solver()
        self._initialized = False

    def _setup_solver(self):

        u_in, p_in, b_in = split(self.xn)

        w, phi = TestFunctions(self._Wup)
        u, p = TrialFunctions(self._Wup)

        k = self.k
        b = -dot(k, u) * (self.N ** 2) * self.beta + b_in

        bcs = [DirichletBC(self._Wup.sub(0), 0.0, "bottom"),
               DirichletBC(self._Wup.sub(0), 0.0, "top")]

        eqn = (
            inner(w, u - u_in)*dx - self.beta*div(w)*p*dx
            - self.beta*inner(w, k)*b*dx
            + phi*(p - p_in)*dx + (self.c ** 2)*self.beta*phi*div(u)*dx
        )

        up_problem = LinearVariationalProblem(lhs(eqn), rhs(eqn),
                                              self.up, bcs=bcs)

        up_solver = LinearVariationalSolver(up_problem,
                                            solver_parameters=self.params)
        self.up_solver = up_solver

        btest = TestFunction(self._Wb)
        btrial = TrialFunction(self._Wb)
        u, p = self.up.split()
        b_eqn = btest*(btrial - b_in + dot(k, u)*(self.N ** 2)*self.beta)*dx
        b_problem = LinearVariationalProblem(lhs(b_eqn), rhs(b_eqn), self.b)
        b_solver = LinearVariationalSolver(b_problem,
                                           solver_parameters=self.b_params)
        self.b_solver = b_solver

    def initialize(self, u_in, p_in, b_in):
        u, p, b = self.xn.split()
        u.assign(u_in)
        p.assign(p_in)
        b.assign(b_in)
        self._initialized = True

    def solve(self):
        if not self._initialized:
            raise RuntimeError("Need initial conditions")

        self.up_solver.solve()

        u, p, b = self.xn.split()
        u1, p1 = self.up.split()
        u.assign(u1)
        p.assign(p1)

        self.b_solver.solve()
        b.assign(self.b)
