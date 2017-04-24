from __future__ import absolute_import, print_function, division

from firedrake import *

from poisson import run_primal_poisson, run_mixed_poisson
from helmholtz import MixedHelmholtzProblem
from meshes import generate_2d_square_mesh


def run_helmholtz_demo(r, d, quads=False):
    """Runs a 2D Helmholtz problem on a unit square domain.
    The solver uses Firedrake's hybridization framework to
    generate the velocity and pressure solutions.

    :arg r: An ``int`` for computing the mesh resolution.
    :arg d: An ``int`` denoting the degree of approximation.
    :arg quads: A ``bool`` specifying whether to use a quad mesh.

    Returns: The velocity and pressure approximations, as well as
             norms describing the approximation error with the
             analytic solutions.
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

test_helmholtz()


# for quad in [False, True]:

#     primal_u = run_primal_poisson(3, 1, quads=quad)
#     sigma_h, u_h, sigma_nh, u_nh, err_s, err_u = run_mixed_poisson(3, 1, quads=quad)

#     name_primal = "primal_poisson"
#     name_hybrid = "hybrid_poisson"
#     name_mixed = "mixed_poisson"

#     if quad:
#         name_primal += "-quad"
#         name_hybrid += "-quad"
#         name_mixed += "-quad"

#     File(name_primal + ".pvd").write(primal_u)
#     File(name_hybrid + ".pvd").write(sigma_h, u_h)
#     File(name_mixed + ".pvd").write(sigma_nh, u_nh)

#     print(err_s)
#     print(err_u)
