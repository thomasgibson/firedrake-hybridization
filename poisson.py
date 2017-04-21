from __future__ import absolute_import, print_function, division

from firedrake import *


def run_primal_poisson(r, d, quads=False):
    """Runs a 3D elliptic solver for the Poisson equation. This
    solves the non-mixed primal Poisson equation using an algebraic
    multigrid solver.

    :arg r: An ``int`` for computing the mesh resolution.
    :arg d: An ``int`` denoting the degree of approximation.
    :arg quads: A ``bool`` specifying whether to use a quad mesh.

    Returns: The scalar solution.
    """

    base = UnitSquareMesh(2 ** r, 2 ** r, quadrilateral=quads)
    layers = 2 ** r
    mesh = ExtrudedMesh(base, layers, layer_height=1.0 / layers)

    V = FunctionSpace(mesh, "CG", d)

    bcs = [DirichletBC(V, 0, "bottom"), DirichletBC(V, 42, "top")]

    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx

    f = Function(V)
    f.assign(0.0)
    L = f * v * dx

    uh = Function(V)
    params = {"pc_type": "hypre",
              "pc_hypre_type": "boomeramg"}
    solve(a == L, uh, bcs=bcs, solver_parameters=params)

    return uh
