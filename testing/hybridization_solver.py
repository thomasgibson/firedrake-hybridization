from __future__ import absolute_import, print_function, division

from firedrake import *

qflag = False
degree = 1

mesh = UnitSquareMesh(2, 2, quadrilateral=qflag)
n = FacetNormal(mesh)

if qflag:
    RT = FiniteElement("RTCF", quadrilateral, degree)
    DG = FiniteElement("DQ", quadrilateral, degree - 1)
    Te = FiniteElement("HDiv Trace", quadrilateral, degree - 1)

else:
    RT = FiniteElement("RT", triangle, degree)
    DG = FiniteElement("DG", triangle, degree - 1)
    Te = FiniteElement("HDiv Trace", triangle, degree - 1)

V = FunctionSpace(mesh, RT)
U = FunctionSpace(mesh, DG)
T = FunctionSpace(mesh, Te)

W = V * U #* T

u, p = TrialFunctions(W)
v, q = TestFunctions(W)

a = (dot(u, v) + div(v)*p + q*div(u))*dx # + (lambdar('+')*jump(v, n=n) + gammar('+')*jump(u, n=n))*dS

x = SpatialCoordinate(mesh)
f = Function(U).assign(0)

L = -f*q*dx + 42*dot(v, n)*ds(4)

bcs = [DirichletBC(W.sub(0), Expression(("0", "0")), (1, 2)),
       # DirichletBC(W.sub(2), Constant(0), (1, 2, 3, 4))]
       ]

params = {"ksp_type": "preonly",
          "ksp_monitor": True,
          "mat_type": "matfree",
          "pc_type": "python",
          "pc_python_type": "firedrake.HybridizationPC",
          "hybridization_pc_type": "lu",
          "hybridization_ksp_type": "preonly"}
# params = {"ksp_type": "gmres",
#           "ksp_rtol": 1e-10}
w = Function(W)
solve(a == L, w, bcs=bcs, solver_parameters=params)

udat, pdat = w.split()
uh = Function(V, name="Approximate flux").assign(udat)
ph = Function(U, name="Approximate scalar").assign(pdat)

x = SpatialCoordinate(mesh)
pp = Function(U, name="Analytic scalar").interpolate(42*x[1])
uu = Function(V, name="Analytic flux").project(grad(42*x[1]))

File("testing.pvd").write(uh, ph, pp, uu)
