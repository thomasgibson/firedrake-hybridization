from __future__ import absolute_import, print_function, division

from firedrake import *

r = 32
mesh = UnitSquareMesh(r, r)
V = FunctionSpace(mesh, "CG", 1)
V_ho = FunctionSpace(mesh, "CG", 5)

bcs = [DirichletBC(V_ho, Constant(0.5), (1, 3)),
       DirichletBC(V_ho, Constant(-0.5), (2, 4))]
expr = Expression('cos(x[0]*pi*2)*sin(x[1]*pi*2)')

v = Function(V)
v.interpolate(expr)

vhbc = Function(V_ho, name="With manual solve")
p = TrialFunction(V_ho)
q = TestFunction(V_ho)
a = inner(p, q)*dx
f = Function(V_ho).interpolate(expr)
L = inner(f, q)*dx
solve(a == L, vhbc, bcs=bcs, solver_parameters={"ksp_type": "preonly",
                                                "pc_type": "lu"})

vp = Function(V_ho, name="With manual bcs applied").interpolate(expr)
for bc in bcs:
    bc.apply(vp)

vpbc = Function(V_ho, name="With projector")
proj = Projector(v, vpbc, bcs=bcs, solver_parameters={"ksp_type": "preonly",
                                                      "pc_type": "lu"}).project()

print(errornorm(vpbc, vhbc))
print(errornorm(vp, vhbc))

File("projection-test.pvd").write(vhbc, vpbc, vp)
