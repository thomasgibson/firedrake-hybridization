from __future__ import absolute_import, print_function, division

from firedrake import *

from poisson import PrimalPoissonProblem, MixedPoissonProblem
from helmholtz import MixedHelmholtzProblem
from meshes import generate_2d_square_mesh, generate_3d_cube_extr_mesh


def run_helmholtz_demo(r, d, quads=False):
    """Runs a 2D Helmholtz problem on a unit square domain.
    The solver uses Firedrake's hybridization framework to
    generate the velocity and pressure solutions.

    :arg r: An ``int`` for computing the mesh resolution.
    :arg d: An ``int`` denoting the degree of approximation.
    :arg quads: A ``bool`` specifying whether to use a quad mesh.

    Returns: The velocity and pressure approximations, as well as
             the analytic solutions.
    """

    mesh = generate_2d_square_mesh(r, quadrilateral=quads)
    helmholtz_problem = MixedHelmholtzProblem(mesh, d)

    params = {"mat_type": "matfree",
              "pc_type": "python",
              "pc_python_type": "firedrake.HybridizationPC",
              "hybridization_ksp_rtol": 1e-8,
              "hybridization_pc_type": "lu",
              "hybridization_ksp_type": "preonly",
              "hybridization_projector_tolerance": 1e-14}
    u, p = helmholtz_problem.solve(params)
    analytic_u, analytic_p = helmholtz_problem.analytic_solution()

    return u, p, analytic_u, analytic_p


def run_primal_poisson_demo(r, d, quads=False):
    """Runs a 3D elliptic solver for the Poisson equation. This
    solves the non-mixed primal Poisson equation using an algebraic
    multigrid solver.

    :arg r: An ``int`` for computing the mesh resolution.
    :arg d: An ``int`` denoting the degree of approximation.
    :arg quads: A ``bool`` specifying whether to use a quad mesh.

    Returns: The scalar approximation and the analytic solution.
    """

    mesh = generate_3d_cube_extr_mesh(r, quadrilateral=quads)
    primal_poisson_problem = PrimalPoissonProblem(mesh, d)

    params = {"pc_type": "hypre",
              "pc_hypre_type": "boomeramg"}
    u = primal_poisson_problem.solve(params)
    analytic_u = primal_poisson_problem.analytic_solution()

    return u, analytic_u


def run_mixed_poisson_demo(r, d, quads=False, hybridize=False):
    """Solves the mixed Poisson equation with strong boundary
    conditions on the scalar unknown. This condition arises in
    the variational form as a natural condition.

    :arg r: An ``int`` for computing the mesh resolution.
    :arg d: An ``int`` denoting the degree of approximation.
    :arg quads: A ``bool`` specifying whether to use a quad mesh.
    :arg hybridize: A ``bool`` indicating whether to use hybridization.

    Returns: The scalar solution and its negative flux for both
             the hybridized case and standard mixed solve. The
             error norms between the two are also returned for
             sanity checking.
    """

    mesh = generate_3d_cube_extr_mesh(r, quadrilateral=quads)
    mixed_poisson_problem = MixedPoissonProblem(mesh, d)

    if hybridize:
        params = {"mat_type": "matfree",
                  "pc_type": "python",
                  "pc_python_type": "firedrake.HybridizationPC",
                  "hybridization_pc_type": "hypre",
                  "hybridization_pc_hypre_type": "boomeramg",
                  "hybridization_ksp_type": "preonly",
                  "hybridization_ksp_rtol": 1e-14}
    else:
        params = {"pc_type": "fieldsplit",
                  "pc_fieldsplit_type": "schur",
                  "ksp_type": "gmres",
                  "pc_fieldsplit_schur_fact_type": "FULL",
                  "fieldsplit_0_ksp_type": "cg",
                  "fieldsplit_0_pc_factor_shift_type": "INBLOCKS",
                  "fieldsplit_1_pc_factor_shift_type": "INBLOCKS",
                  "fieldsplit_1_ksp_type": "cg"}
    u, p = mixed_poisson_problem.solve(params)
    analytic_u, analytic_p = mixed_poisson_problem.analytic_solution()

    return u, p, analytic_u, analytic_p


def test_helmholtz():
    for quad in [False, True]:
        u, p, analytic_u, analytic_p = run_helmholtz_demo(6, 1, quads=quad)
        u_err = errornorm(u, analytic_u)
        p_err = errornorm(p, analytic_p)

        name = "mixed_helmholtz"
        if quad:
            name += "-quad"

        print(u_err)
        print(p_err)
        File(name + ".pvd").write(u, p, analytic_u, analytic_p)


def test_primal_poisson():
    for quad in [False, True]:
        u, analytic_u = run_primal_poisson_demo(6, 1, quads=quad)
        u_err = errornorm(u, analytic_u)

        name = "primal_poisson"
        if quad:
            name += "-quad"

        print(u_err)
        File(name + ".pvd").write(u, analytic_u)


def test_mixed_poisson(hybridize=False):
    for quad in [False, True]:
        u, p, analytic_u, analytic_p = run_mixed_poisson_demo(5, 1, quads=quad,
                                                              hybridize=hybridize)
        u_err = errornorm(u, analytic_u)
        p_err = errornorm(p, analytic_p)

        name = "mixed_poisson"
        if quad:
            name += "-quad"

        print(u_err)
        print(p_err)
        File(name + ".pvd").write(u, p, analytic_u, analytic_p)

# test_primal_poisson()
# test_helmholtz()
flag = True
test_mixed_poisson(hybridize=flag)
