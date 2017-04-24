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

        self._linear_form = -20.0*v*dx + 10.0*v*ds_tb

        x = SpatialCoordinate(mesh)
        bc_fct = 10.0*(x[2] - 0.5)*(x[2] - 0.5)

        self._strong_bcs = [DirichletBC(self._H1_space, bc_fct, 1),
                            DirichletBC(self._H1_space, bc_fct, 2),
                            DirichletBC(self._H1_space, bc_fct, 3),
                            DirichletBC(self._H1_space, bc_fct, 4)]

        analytic_sol = Function(self._H1_space, name="Analytic scalar")
        analytic_sol.interpolate(bc_fct)
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

        u = Function(self._H1_space, name="Approximate scalar")
        u.assign(uh)

        return u


class MixedPoissonProblem(object):
    """A class describing the mixed Poisson problem with
    strong boundary conditions on the scalar variable. The
    boundary condition arises in the variational formulation
    as a natural boundary condition.

    This operator uses the classical mixed Raviart-Thomas formulation
    for both simplicial and hexahedral extruded meshes.
    """

    def __init__(self, mesh, degree):
        """Constructor for the MixedPoissonProblem class.

        :arg mesh: An extruded Firedrake mesh.
        :arg degree: The degree of approximation.
        """

        super(MixedPoissonProblem, self).__init__()

        if not mesh.cell_set._extruded:
            raise ValueError("This problem is designed for an extruded mesh.")

        n = FacetNormal(mesh)

        if mesh._base_mesh.ufl_cell() == quadrilateral:
            RT = FiniteElement("RTCF", quadrilateral, degree)
            DG_v = FiniteElement("DG", interval, degree - 1)
            DG_h = FiniteElement("DQ", quadrilateral, degree - 1)
            CG = FiniteElement("CG", interval, degree)
            U = FunctionSpace(mesh, "DQ", degree - 1)

        else:
            RT = FiniteElement("RT", triangle, degree)
            DG_v = FiniteElement("DG", interval, degree - 1)
            DG_h = FiniteElement("DG", triangle, degree - 1)
            CG = FiniteElement("CG", interval, degree)
            U = FunctionSpace(mesh, "DG", degree - 1)

        HDiv_ele = EnrichedElement(HDiv(TensorProductElement(RT, DG_v)),
                                   HDiv(TensorProductElement(DG_h, CG)))
        V = FunctionSpace(mesh, HDiv_ele)
        W = V * U
        self._mixedspace = W
        self._hdiv_space = V
        self._L2_space = U

        self._trial_functions = TrialFunctions(self._mixedspace)
        self._test_functions = TestFunctions(self._mixedspace)

        u, p = self._trial_functions
        v, q = self._test_functions
        self._bilinear_form = (dot(u, v) + div(v)*p + q*div(u))*dx

        x = SpatialCoordinate(mesh)
        bc_fct = 10.0*(x[2] - 0.5)*(x[2] - 0.5)
        g = Function(self._L2_space)
        g.interpolate(bc_fct)

        self._linear_form = -20.0*q*dx + g*dot(v, n)*ds_v

        bc0 = DirichletBC(W.sub(0), Constant((0.0, 0.0, 10.0)), "top")
        bc1 = DirichletBC(W.sub(0), Constant((0.0, 0.0, 10.0)), "bottom")
        self._bcs = [bc0, bc1]

        analytic_scalar = Function(self._L2_space, name="Analytic scalar")
        analytic_scalar.interpolate(bc_fct)
        analytic_flux = Function(self._hdiv_space, name="Analytic flux")
        analytic_flux.project(grad(bc_fct))
        self._analytic_solution = (analytic_flux, analytic_scalar)

    def analytic_solution(self):
        """Returns the analytic solution of the problem."""
        return self._analytic_solution

    def solve(self, parameters):
        """Solves the mixed Poisson problem given a set
        of solver parameters.

        :arg parameters: A ``dict`` of solver parameters.
        """

        w = Function(self._mixedspace)
        solve(self._bilinear_form == self._linear_form, w,
              bcs=self._bcs,
              solver_parameters=parameters)
        udat, pdat = w.split()

        u = Function(self._hdiv_space, name="Approximate flux")
        p = Function(self._L2_space, name="Approximate scalar")
        u.assign(udat)
        p.assign(pdat)

        return u, p
