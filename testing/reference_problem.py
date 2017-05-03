from __future__ import absolute_import, print_function, division

from firedrake import *

qflag = False
degree = 1
hybrid = True

mesh = UnitSquareMesh(2, 2, quadrilateral=qflag)
n = FacetNormal(mesh)

if qflag:
    RT = FiniteElement("RTCF", quadrilateral, degree)
    DG = FiniteElement("DQ", quadrilateral, degree - 1)

else:
    RT = FiniteElement("RT", triangle, degree)
    DG = FiniteElement("DG", triangle, degree - 1)

V = FunctionSpace(mesh, RT)
U = FunctionSpace(mesh, DG)

W = V * U

u, p = TrialFunctions(W)
v, q = TestFunctions(W)

a = (dot(u, v) + div(v)*p + q*div(u))*dx

x = SpatialCoordinate(mesh)
f = Function(U).assign(0)

L = -f*q*dx + 42*dot(v, n)*ds(4)

bcs = [DirichletBC(W.sub(0), Expression(("0", "0")), (1, 2))]

if hybrid:
    params = {"ksp_type": "preonly",
              "mat_type": "matfree",
              "pc_type": "python",
              "pc_python_type": "firedrake.HybridizationPC",
              "hybridization_ksp_type": "preonly",
              "hybridization_pc_type": "lu",
              "hybridization_projector_tolerance": 1e-10}
else:
    params = {"ksp_type": "gmres",
              "ksp_rtol": 1e-10}

w = Function(W)
solve(a == L, w, bcs=bcs, solver_parameters=params)
