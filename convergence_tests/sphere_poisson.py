from __future__ import absolute_import, division
from firedrake import *
import numpy as np


def poisson_sphere(MeshClass, refinement, hdiv_space):
    """Test hybridizing lowest order mixed methods on a sphere."""
    mesh = MeshClass(refinement_level=refinement)
    mesh.init_cell_orientations(Expression(("x[0]", "x[1]", "x[2]")))
    x, y, z = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, hdiv_space, 1)
    U = FunctionSpace(mesh, "DG", 0)
    W = U * V

    f = Function(U)
    f.interpolate(x*y*z)

    u_exact = Function(U).interpolate(x*y*z/12.0)

    u, sigma = TrialFunctions(W)
    v, tau = TestFunctions(W)

    a = (dot(sigma, tau) - div(tau)*u + v*div(sigma))*dx
    L = f*v*dx
    w = Function(W)

    nullsp = MixedVectorSpaceBasis(W, [VectorSpaceBasis(constant=True), W[1]])
    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'preonly',
                                'pc_type': 'lu',
                                'pc_factor_mat_solver_package': 'mumps',
                                'hdiv_residual': {'ksp_type': 'cg',
                                                  'ksp_rtol': 1e-14,
                                                  'pc_type': 'bjacobi',
                                                  'sub_pc_type': 'ilu'},
                                'use_reconstructor': True}}
    solve(a == L, w, nullspace=nullsp, solver_parameters=params)
    u_h, _ = w.split()
    error = errornorm(u_exact, u_h)
    return w, error

MeshClass = UnitCubedSphereMesh
refinement = 4
hdiv_family = "RTCF"

w, err = poisson_sphere(MeshClass, refinement, hdiv_family)
print err
File("poisson_sphere.pvd").write(w.split()[0], w.split()[1])
