from __future__ import absolute_import, print_function, division

from firedrake import *


class MixedHelmholtzProblem(object):
    """A class describing the mixed Helmholtz problem.

    This operator uses the classical mixed Raviart-Thomas formulation
    for both simplicial and quadrilateral 2D meshes.
    """

    def __init__(self, mesh, degree):
        """Constructor for the MixedHelmholtzProblem class.

        :arg mesh: A firedrake mesh.
        :arg degree: The degree of approximation.
        """

        super(MixedHelmholtzProblem, self).__init__()

        if mesh.ufl_cell() == quadrilateral:
            V = FunctionSpace(mesh, "RTCF", degree)
            U = FunctionSpace(mesh, "DQ", degree - 1)
        else:
            V = FunctionSpace(mesh, "RT", degree)
            U = FunctionSpace(mesh, "DG", degree - 1)

        self._mixedspace = V * U
        self._hdiv_space = V
        self._L2_space = U

        self._trial_functions = TrialFunctions(self._mixedspace)
        self._test_functions = TestFunctions(self._mixedspace)

        u, p = self._trial_functions
        v, q = self._test_functions
        self._bilinear_form = (dot(u, v) - div(v)*p + q*div(u) + p*q)*dx

        forcing_function = Function(self._L2_space)
        x, y = SpatialCoordinate(mesh)
        forcing_function.interpolate((1 + 8*pi*pi)*sin(2*pi*x)*sin(2*pi*y))
        self._f = forcing_function

        self._linear_form = self._f*q*dx

        analytic_scalar = Function(self._L2_space,
                                   name="Analytic scalar (mixed)")
        analytic_scalar.interpolate(sin(2*pi*x)*sin(2*pi*y))
        analytic_flux = Function(self._hdiv_space,
                                 name="Analytic flux (mixed)")
        analytic_flux.project(-grad(sin(2*pi*x)*sin(2*pi*y)))
        self._analytic_solution = (analytic_flux, analytic_scalar)

    def analytic_solution(self):
        """Returns the analytic solution of the problem."""
        return self._analytic_solution

    def solve(self, parameters):
        """Solves the mixed Helmholtz problem given a set
        of solver parameters.

        :arg parameters: A ``dict`` of solver parameters.
        """

        w = Function(self._mixedspace)
        solve(self._bilinear_form == self._linear_form, w,
              solver_parameters=parameters)
        udat, pdat = w.split()

        u = Function(self._hdiv_space, name="Approximate flux (mixed)")
        p = Function(self._L2_space, name="Approximate scalar (mixed)")
        u.assign(udat)
        p.assign(pdat)

        return u, p
