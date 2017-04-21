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


def run_hybrid_poisson(r, d, quads=False):
    """Runs a 3D solver for the mixed Poisson equation. This
    solves the Poisson equation using a hybrided mixed method
    solver.

    :arg r: An ``int`` for computing the mesh resolution.
    :arg d: An ``int`` denoting the degree of approximation.
    :arg quads: A ``bool`` specifying whether to use a quad mesh.

    Returns: The scalar solution.
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

    f = Function(U)
    f.assign(0.0)
    L = -42.0 * dot(tau, n) * ds_t

    wh = Function(W)
    params = {"mat_type": "matfree",
              "pc_type": "python",
              "pc_python_type": "firedrake.HybridizationPC",
              "hybridization_pc_type": "hypre",
              "hybridization_pc_hypre_type": "boomeramg",
              "hybridization_ksp_rtol": 1e-8,
              "hybridization_projector_tolerance": 1e-14}
    solve(a == L, wh, solver_parameters=params)
    sigmah, uh = wh.split()

    return sigmah, uh
