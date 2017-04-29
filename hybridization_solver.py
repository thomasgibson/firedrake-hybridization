from __future__ import absolute_import, print_function, division

from firedrake import *
from firedrake.formmanipulation import split_form

qflag = False
degree = 1

mesh = UnitSquareMesh(32, 32, quadrilateral=qflag)
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
L = f*v*dx - 10*dot(tau, n)*ds(4) - 10*dot(tau, n)*ds(2)

bcs = DirichletBC(T, Constant(0.0), (1, 2, 3, 4))

gammar = TestFunction(T)
trace_form = dot(sigma, n)*gammar('+')*dS

# Forward elimination
K = Tensor(trace_form)
A = Tensor(a)
F = assemble(L)
S = assemble(K * A.inv * K.T, bcs=bcs)
E = assemble(K * A.inv * F)

# Solve for the Lagrange multipliers
lambda_sol = Function(T)
solve(S, lambda_sol, E, solver_parameters={'pc_type': 'lu',
                                           'ksp_type': 'preonly'})

# Backwards reconstruction
split_mixed_op = dict(split_form(A.form))
split_trace_op = dict(split_form(K.form))
split_rhs = F.split()

A = Tensor(split_mixed_op[(0, 0)])
B = Tensor(split_mixed_op[(0, 1)])
C = Tensor(split_mixed_op[(1, 0)])
D = Tensor(split_mixed_op[(1, 1)])
K0 = Tensor(split_trace_op[(0, 0)])
K1 = Tensor(split_trace_op[(0, 1)])
g = split_rhs[0]
f = split_rhs[1]

M = D - C * A.inv * B
R = K1.T - C * A.inv * K0.T
u = Function(U)
u_rec = M.inv*f - M.inv*C*A.inv*g + M.inv*(C*A.inv*K0.T*lambda_sol - K1.T*lambda_sol)
u = assemble(u_rec, tensor=u)

sigma = assemble(A.inv*g - A.inv*B*u - A.inv*K0.T*lambda_sol)

File("hybrid_solution.pvd").write(u, project(sigma, FunctionSpace(mesh, RT)))
