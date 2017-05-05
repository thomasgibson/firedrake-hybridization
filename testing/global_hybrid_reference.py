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

V = FunctionSpace(mesh, BrokenElement(RT))
U = FunctionSpace(mesh, DG)
T = FunctionSpace(mesh, Te)

W = V * U * T

u, p, lambdar = TrialFunctions(W)
v, q, gammar = TestFunctions(W)

a_dx = (dot(u, v) + div(v)*p + q*div(u))*dx
a_dS = (lambdar('+')*jump(v, n=n) + gammar('+')*jump(u, n=n))*dS
a = a_dx + a_dS

x = SpatialCoordinate(mesh)
f = Function(U).assign(0)

L = -f*q*dx + 42*dot(v, n)*ds(4)

bcs = [DirichletBC(W.sub(0), Expression(("0", "0")), (1, 2)),
       DirichletBC(W.sub(2), Constant(0), (1, 2, 3, 4))]

params = {"ksp_type": "gmres",
          "ksp_rtol": 1e-10}
w = Function(W)
solve(a == L, w, bcs=bcs, solver_parameters=params)
uh, ph, _ = w.split()

Vc = FunctionSpace(mesh, RT)
W2 = Vc * U

u, p = TrialFunctions(W2)
v, q = TestFunctions(W2)

a = (dot(u, v) + div(v)*p + q*div(u))*dx

x = SpatialCoordinate(mesh)
f = Function(U).assign(0)
bcs = DirichletBC(W2.sub(0), Expression(("0", "0")), (1, 2))

L = -f*q*dx + 42*dot(v, n)*ds(4)
w2 = Function(W2)
solve(a == L, w2, bcs=bcs, solver_parameters=params)
uc, pc = w2.split()

print(errornorm(project(uh, Vc), uc))
print(errornorm(ph, pc))
