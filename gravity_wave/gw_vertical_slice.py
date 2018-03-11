from firedrake import *
from function_spaces import construct_spaces
from solver import GravityWaveSolver
import numpy as np


def compressible_hydrostatic_pressure(W2v, W3, k, b0, p0,
                                      top=False, params=None):

    # get F
    v = TrialFunction(W2v)
    w = TestFunction(W2v)

    if top:
        bstring = "top"
    else:
        bstring = "bottom"

    bcs = [DirichletBC(W2v, 0.0, bstring)]

    a = inner(w, v)*dx
    L = inner(k, w)*b0*dx
    F = Function(W2v)

    solve(a == L, F, bcs=bcs)

    # define mixed function space
    WV = W2v * W3

    # get pprime
    v, pprime = TrialFunctions(WV)
    w, phi = TestFunctions(WV)

    bcs = [DirichletBC(WV[0], 0.0, bstring)]

    a = (
        inner(w, v) + div(w)*pprime + div(v)*phi + phi*pprime
    )*dx
    L = phi*div(F)*dx
    w1 = Function(WV)

    if params is None:
        params = {'ksp_type': 'gmres',
                  'pc_type': 'fieldsplit',
                  'pc_fieldsplit_type': 'schur',
                  'pc_fieldsplit_schur_fact_type': 'full',
                  'pc_fieldsplit_schur_precondition': 'selfp',
                  'fieldsplit_1_ksp_type': 'preonly',
                  'fieldsplit_1_pc_type': 'gamg',
                  'fieldsplit_1_mg_levels_pc_type': 'bjacobi',
                  'fieldsplit_1_mg_levels_sub_pc_type': 'ilu',
                  'fieldsplit_0_ksp_type': 'richardson',
                  'fieldsplit_0_ksp_max_it': 4,
                  'ksp_atol': 1.e-08,
                  'ksp_rtol': 1.e-08}

    solve(a == L, w1, bcs=bcs, solver_parameters=params)

    v, pprime = w1.split()
    p0.project(pprime)


dt = 6.
tmax = 3600.

columns = 300
L = 3.0e5
base = PeriodicIntervalMesh(columns, L)
nlayers = 10
H = 1.0e4
height = H/nlayers
mesh = ExtrudedMesh(base, layers=nlayers, layer_height=height)

W2, W3, Wb, W2v = construct_spaces(mesh, order=1)
x, z = SpatialCoordinate(mesh)
solver = GravityWaveSolver(mesh, dt, W2, W3, Wb)

N = solver.N
bref = z*(N**2)
b_b = Function(Wb).interpolate(bref)
a = 5.0e3
deltab = 1.0e-2
b_pert = deltab*sin(np.pi*z/H)/(1 + (x - L/2)**2/a**2)

b0 = Function(Wb)
b0.interpolate(b_b + b_pert)

u0 = Function(W2)
u0.project(as_vector([20.0, 0.0]))

p0 = Function(W3)
compressible_hydrostatic_pressure(W2v, W3, solver.k, b0, p0)

solver.initialize(u0, p0, b0)

outfile = File("results/compressible-gw/gw.pvd")
un, pn, bn = solver.xn.split()
outfile.write(un, pn, bn)

t = 0.0
while t < tmax:
    t += dt
    solver.solve()
    outfile.write(un, pn, bn)
