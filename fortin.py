from __future__ import absolute_import, print_function, division

from firedrake import *

mesh = UnitSquareMesh(2, 2)

RT = FiniteElement("RT", triangle, 1)
BRT = BrokenElement(RT)

U = FunctionSpace(mesh, RT)
Ud = FunctionSpace(mesh, BRT)

x = SpatialCoordinate(mesh)
fct = as_vector([x[0] ** 2, x[1] ** 2])
u0 = Function(U).project(fct)
u = Function(Ud).project(fct)

Tr = FunctionSpace(mesh, "HDiv Trace", 0)

gammar = TestFunction(Tr)
lambdar = TrialFunction(Tr)

n = FacetNormal(mesh)

a_dS = lambdar('+')*gammar('+')*dS + lambdar*gammar*ds
l_dS = gammar('+')*0.5*jump(dot(u, n))*dS + gammar*0.5*dot(u, n)*ds
f = Function(Tr)
# A = assemble(a_dS).M.values
# b = assemble(l_dS).dat.data

solve(a_dS == l_dS, f)

ubar = TrialFunction(U)

a = gammar('+')*dot(ubar, n)('+')*dS + gammar*dot(ubar, n)*ds
l = gammar('+')*f('+')*dS + gammar*f*ds

a_global = gammar*jump(ubar, n=n)*dS + gammar*dot(ubar, n)*ds

M = Tensor(a)
q = Tensor(l)
x = assemble(M.inv*q)



ans = Function(U)
# solve(a == l, ans)

count_code = """
for (int i; i<count.ndofs; ++i) {
    count[i][0] += 1.0;
}
"""

kernel_code = """
for (int i; i<ubar.ndofs; ++i) {
    ubar[i][0] += u[i][0]/one[i][0];
}
"""

One = Function(U)

par_loop(count_code, dx, {"count":(One, INC)})
par_loop(kernel_code, dx, {"ubar":(ans, INC), "u":(u, READ), "one":(One, READ)})
