from __future__ import absolute_import, print_function, division

from firedrake import *
from helmholtz import MixedHelmholtzProblem
from meshes import generate_2d_square_mesh

import matplotlib as plt


def run_helmholtz_resolution_test(degree, quadrilateral=False):
    """
    """

    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization_ksp_rtol': 1e-8,
              'hybridization_pc_type': 'lu',
              'hybridization_pc_factor_mat_solver_package': 'mumps',
              'hybridization_ksp_type': 'preonly',
              'hybridization_projector_tolerance': 1e-14}
    scalar_data = []
    flux_data = []
    for r in range(1, 10):
        mesh = generate_2d_square_mesh(r, quadrilateral=quadrilateral)
        problem = MixedHelmholtzProblem(mesh, degree)
        u, p = problem.solve(params)
        analytic_u, analytic_p = problem.analytic_solution()
        err_u = errornorm(u, analytic_u)
        err_p = errornorm(p, analytic_p)
        scalar_data.append(err_p)
        flux_data.append(err_u)

    return scalar_data, flux_data
