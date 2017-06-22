from __future__ import absolute_import, division
from firedrake import *


class HybridMixedHelmholtzProblem(object):
    """
    """

    def __init__(self, degree, resolution, hexes=False):
        """
        """

        # Set up the computational domain: extruded unit cube mesh
        # with same resolution in all spatial directions.
        N = resolution
        x = 2 ** N
        y = 2 ** N
        mesh = UnitSquareMesh(x, y, quadrilateral=hexes)
        self.mesh = mesh

        # Set up the function spaces for the hybrid-mixed formulation of
        # the Helmholtz equation. This includes:
        # (1) A "broken" HDiv space;
        # (2) A DG scalar field space;
        # (3) A space for the approximate traces
        if hexes:
            RT = FiniteElement("RTCF", quadrilateral, degree)
            U = FunctionSpace(mesh, "DQ", degree - 1)
            Ue = FunctionSpace(mesh, "DQ", degree)
        else:
            RT = FiniteElement("RT", triangle, degree)
            U = FunctionSpace(mesh, "DG", degree - 1)
            Ue = FunctionSpace(mesh, "DG", degree)

        HDiv_ele = RT
        V = FunctionSpace(mesh, HDiv_ele)
        Vd = FunctionSpace(mesh, BrokenElement(HDiv_ele))
        T = FunctionSpace(mesh, "HDiv Trace", degree - 1)

        W = V * U
        Wd = Vd * U
        self.W = W
        self.Wd = Wd
        self.T = T

        # Store function for the Lagrange multipliers (for post-processing)
        self.lambdar_sol = Function(T)

        # Store function for broken and nonbroken solution
        self.broken_w = Function(Wd)
        self.w = Function(W)

        # Set up the linear and bilinear forms
        x, y = SpatialCoordinate(mesh)
        f = Function(U)
        expr = (1+8*pi*pi)*sin(2*pi*x)*sin(2*pi*y)
        f.interpolate(expr)

        sigma, u = TrialFunctions(Wd)
        tau, v = TestFunctions(Wd)

        a = dot(sigma, tau)*dx + u*v*dx + div(sigma)*v*dx - div(tau)*u*dx
        L = f*v*dx
        self._bilinear_form = a
        self._linear_form = L

        gammar = TestFunction(T)
        n = FacetNormal(mesh)
        sigma = TrialFunctions(Wd)[0]
        trace_form = gammar('+') * dot(sigma, n) * dS
        self.trace_form = trace_form

        # Store trace bcs
        bcs = [DirichletBC(T, Constant(0.0), "on_boundary")]
        self.trace_bcs = bcs

        # Store the exact solution of the problem:
        u_exact = Function(Ue).interpolate(sin(2*pi*x)*sin(2*pi*y))
        self.u_exact = u_exact

    @property
    def a(self):
        """
        """
        return self._bilinear_form

    @property
    def multiplier_integrals(self):
        """
        """
        return self.trace_form

    @property
    def L(self):
        """
        """
        return self._linear_form

    @property
    def exact_solution(self):
        """
        """
        return self.u_exact
