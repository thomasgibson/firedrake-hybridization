from __future__ import absolute_import, print_function, division

from firedrake import *

qflag = False
degree = 1

mesh = UnitSquareMesh(8, 8, quadrilateral=qflag)
n = FacetNormal(mesh)

if qflag:
    RT = FiniteElement("RTCF", quadrilateral, degree)
    DG = FiniteElement("DQ", quadrilateral, degree - 1)
    Te = FiniteElement("HDiv Trace", quadrilateral, degree - 1)

else:
    RT = FiniteElement("RT", triangle, degree)
    DG = FiniteElement("DG", triangle, degree - 1)
    Te = FiniteElement("HDiv Trace", triangle, degree - 1)

Vd = FunctionSpace(mesh, BrokenElement(RT))
U = FunctionSpace(mesh, DG)
T = FunctionSpace(mesh, Te)

W = Vd * U

sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)

f = Function(U)
x, y = SpatialCoordinate(mesh)
f.interpolate((1 + 8*pi*pi)*sin(2*pi*x)*sin(2*pi*y))

a = (dot(sigma, tau) - div(tau)*u + v*div(sigma) + u*v)*dx
L = f*v*dx

bcs = DirichletBC(T, Constant(0.0), (1, 2, 3, 4))

gammar = TestFunction(T)
trace_form = dot(sigma, n)*gammar('+')*dS

K = Tensor(trace_form)
A = Tensor(a)
F = Tensor(L)

S = assemble(K * A.inv * K.T, bcs=bcs)
E = assemble(K * A.inv * F)

lambda_sol = Function(T)
solve(S, lambda_sol, E, solver_parameters={'pc_type': 'lu',
                                           'ksp_type': 'preonly'})
