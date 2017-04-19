from __future__ import absolute_import, print_function, division

from firedrake import *


def run_helmholtz(r, d, quads=False):
    """Runs a 2D Helmholtz problem on a unit square domain.
    The solver uses Firedrake's hybridization framework to
    generate the velocity and pressure solutions.

    :arg r: An ``int`` for computing the mesh resolution.
    :arg d: An ``int`` denoting the degree of approximation.
    :arg quads: A ``bool`` specifying whether to use a quad mesh.

    Returns: The velocity and pressure approximations.
    """

    mesh = UnitSquareMesh(2 ** r, 2 ** r, quadrilateral=quads)

    if quads:
        V = FunctionSpace(mesh, "RTCF", d)
        U = FunctionSpace(mesh, "DQ", d - 1)
    else:
        V = FunctionSpace(mesh, "RT", d)
        U = FunctionSpace(mesh, "DG", d - 1)

    W = V * U
    w = Function(W)

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    a = (dot(u, v) - div(v) * p + q * div(u) + p * q) * dx

    f = Function(U)
    x, y = SpatialCoordinate(mesh)
    f.interpolate((1 + 8*pi*pi)*sin(2*pi*x)*sin(2*pi*y))

    L = f * q * dx

    params = {"mat_type": "matfree",
              "pc_type": "python",
              "pc_python_type": "firedrake.HybridizationPC",
              "hybridization_ksp_rtol": 1e-8,
              "hybridization_pc_type": "lu",
              "hybridization_ksp_type": "preonly",
              "hybridization_projector_tolerance": 1e-14}

    solve(a == L, w, solver_parameters=params)
    vel, pr = w.split()

    return vel, pr
