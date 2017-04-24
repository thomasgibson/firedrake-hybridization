from __future__ import absolute_import, print_function, division

from firedrake import *


class PrimalPoissonProblem(object):
    """A class describing the primal Poisson problem with
    strong boundary conditions.

    This operator uses the classical H1 formulation of the problem
    for both simplicial and hexahedral extruded meshes.
    """

    def __init__(self, mesh, degree):
        """Constructor for the PrimalPoissonProblem class.

        :arg mesh: An extruded Firedrake mesh.
        :arg degree: The degree of approximation.
        """

        super(PrimalPoissonProblem, self).__init__()

        if not mesh.cell_set._extruded:
            raise ValueError("This problem is designed for an extruded mesh.")

        V = FunctionSpace(mesh, "CG", degree)

        self._H1_space = V

        self._trial_function = TrialFunction(self._H1_space)
        self._test_function = TestFunction(self._H1_space)

        u = self._trial_function
        v = self._test_function
        self._bilinear_form = inner(grad(u), grad(v))*dx

        # Firedrake requires a linear form, even if it's just 0
        f = Function(self._H1_space)
        f.assign(0.0)
        self._linear_form = f*v*dx

        self._strong_bcs = [DirichletBC(self._H1_space, 0.0, "bottom"),
                            DirichletBC(self._H1_space, 42.0, "top")]

        analytic_sol = Function(self._H1_space)
        x = SpatialCoordinate(mesh)
        analytic_sol.interpolate(42.0*x[2])
        self._analytic_solution = analytic_sol

    def analytic_solution(self):
        """Returns the analytic solution of the problem."""
        return self._analytic_solution

    def solve(self, parameters):
        """Solves the primal Poisson equation given a set
        of solver parameters.

        :arg parameters: A ``dict`` of solver parameters.
        """

        uh = Function(self._H1_space)
        solve(self._bilinear_form == self._linear_form, uh,
              bcs=self._strong_bcs,
              solver_parameters=parameters)
        return uh


def run_mixed_poisson(r, d, quads=False):
    """Solves the mixed Poisson equation with strong boundary
    conditions on the scalar unknown. This condition arises in
    the variational form as a natural condition.

    A hybridized and non-hybridized approach is taken. The solver
    parameters specify which technique is used.

    :arg r: An ``int`` for computing the mesh resolution.
    :arg d: An ``int`` denoting the degree of approximation.
    :arg quads: A ``bool`` specifying whether to use a quad mesh.

    Returns: The scalar solution and its negative flux for both
             the hybridized case and standard mixed solve. The
             error norms between the two are also returned for
             sanity checking.
    """

    base = UnitSquareMesh(2 ** r, 2 ** r, quadrilateral=quads)
    layers = 2 ** r
    mesh = ExtrudedMesh(base, layers, layer_height=1.0 / layers)
    n = FacetNormal(mesh)

    if quads:
        RT = FiniteElement("RTCF", quadrilateral, d)
        DG_v = FiniteElement("DG", interval, d - 1)
        DG_h = FiniteElement("DQ", quadrilateral, d - 1)
        CG = FiniteElement("CG", interval, d)
        U = FunctionSpace(mesh, "DQ", d - 1)

    else:
        RT = FiniteElement("RT", triangle, d)
        DG_v = FiniteElement("DG", interval, d - 1)
        DG_h = FiniteElement("DG", triangle, d - 1)
        CG = FiniteElement("CG", interval, d)
        U = FunctionSpace(mesh, "DG", d - 1)

    HDiv_ele = EnrichedElement(HDiv(TensorProductElement(RT, DG_v)),
                               HDiv(TensorProductElement(DG_h, CG)))
    V = FunctionSpace(mesh, HDiv_ele)
    W = V * U

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    a = (dot(sigma, tau) - div(tau) * u + div(sigma) * v) * dx

    L = -42.0 * dot(tau, n) * ds_t

    params = {"mat_type": "matfree",
              "pc_type": "python",
              "pc_python_type": "firedrake.HybridizationPC",
              "hybridization_pc_type": "hypre",
              "hybridization_pc_hypre_type": "boomeramg",
              "hybridization_ksp_type": "preonly",
              "hybridization_ksp_rtol": 1e-14}

    params2 = {"pc_type": "fieldsplit",
               "pc_fieldsplit_type": "schur",
               "ksp_type": "gmres",
               "pc_fieldsplit_schur_fact_type": "FULL",
               "fieldsplit_0_ksp_type": "cg",
               "fieldsplit_0_pc_factor_shift_type": "INBLOCKS",
               "fieldsplit_1_pc_factor_shift_type": "INBLOCKS",
               "fieldsplit_1_ksp_type": "cg"}

    wh = Function(W)
    solve(a == L, wh, solver_parameters=params)
    sigma_h, u_h = wh.split()

    w = Function(W)
    solve(a == L, w, solver_parameters=params2)
    sigma_nh, u_nh = w.split()

    return (sigma_h, u_h, sigma_nh, u_nh,
            errornorm(sigma_h, sigma_nh),
            errornorm(u_h, u_nh))
