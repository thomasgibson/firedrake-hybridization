from __future__ import absolute_import, print_function, division

from firedrake import *

mesh = UnitSquareMesh(2, 2)
RT = FiniteElement("RT", triangle, 1)
V = FunctionSpace(mesh, RT)
u = TrialFunction(V)
v = TestFunction(V)

f = Function(V)
x = SpatialCoordinate(mesh)
f.project(as_vector([x[1], x[0]]))

r = Function(V)
a = inner(u, v)*dx
L = inner(f, v)*dx
solve(a == L, r)

V_d = FunctionSpace(mesh, BrokenElement(RT))
phi_d = TestFunction(V_d)

r_d = assemble(inner(r, phi_d)*dx)

ref = assemble(inner(f, phi_d)*dx)

print(errornorm(r_d, ref))
