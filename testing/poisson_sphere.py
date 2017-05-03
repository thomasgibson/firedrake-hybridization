from __future__ import absolute_import, print_function, division

from firedrake import *

mesh = UnitIcosahedralSphereMesh(refinement_level=3, degree=3)
mesh.init_cell_orientations(Expression(("x[0]", "x[1]", "x[2]")))
x, y, z = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "BDM", 2)
U = FunctionSpace(mesh, "DG", 1)
W = V * U

f = Function(U)
f.interpolate(x*y*z)

u_exact = Function(U, name="exact scalar").interpolate(x*y*z/12.0)

sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)

a = (dot(sigma, tau) - div(tau)*u + v*div(sigma))*dx
L = f*v*dx

params = {"ksp_type": "preonly",
          "ksp_monitor": True,
          "mat_type": "matfree",
          "pc_type": "python",
          "pc_python_type": "firedrake.HybridizationPC",
          "hybridization_pc_type": "lu",
          "hybridization_ksp_type": "preonly",
          "hybridization_projector_tolerance": 1e-14}

nullsp = MixedVectorSpaceBasis(W, [W[0], VectorSpaceBasis(constant=True)])
w = Function(W)
solve(a == L, w, nullspace=nullsp, solver_parameters=params)

vdat, pdat = w.split()

File("sphere.pvd").write(vdat, pdat, u_exact)
