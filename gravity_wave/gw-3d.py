from firedrake import *
from function_spaces import construct_spaces
from solver import GravityWaveSolver
from gaussian import MultipleGaussianExpression
import math
import numpy as np


# Spherical shell (Earth with layers)
# Number of vertical layers
nlayer = 40
# Radius of earth
r_earth = 6.371e6/125.0
# Thickness of spherical shell
thickness = 1.0e4

order = 1

base = IcosahedralSphereMesh(r_earth,
                             refinement_level=5,
                             degree=3)
mesh = ExtrudedMesh(base, extrusion_type='radial',
                    layers=nlayer, layer_height=thickness/nlayer)
num_cells = mesh.cell_set.size
solver_type = "AMG"

g = MultipleGaussianExpression(1, r_earth, thickness)
expression = Expression(str(g))

W2, W3, Wb, _ = construct_spaces(mesh, order=order)

u0 = Function(W2)
x = SpatialCoordinate(mesh)
u_max = 20.
uexpr = as_vector([-u_max*x[1]/r_earth,
                   u_max*x[0]/r_earth, 0.0])
u0.project(uexpr)

lamda_c = 2.0*np.pi/3.0
phi_c = 0.0
W_CG1 = FunctionSpace(mesh, "CG", 1)
z = Function(W_CG1).interpolate(Expression("sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]) - a",
                                           a=r_earth))
lat = Function(W_CG1).interpolate(Expression("asin(x[2]/sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]))"))
lon = Function(W_CG1).interpolate(Expression("atan2(x[1], x[0])"))
b0 = Function(Wb)
deltaTheta = 1.0
L_z = 20000.0
d = 5000.0

sin_tmp = sin(lat) * sin(phi_c)
cos_tmp = cos(lat) * cos(phi_c)

r = r_earth*acos(sin_tmp + cos_tmp*cos(lon-lamda_c))
s = (d**2)/(d**2 + r**2)

bexpr = deltaTheta*s*sin(2*np.pi*z/L_z)
b0.interpolate(bexpr)

p0 = Function(W3).assign(0.0)

c = 300
N = 0.01
Omega = 7.292e-5
nu_cfl = 8.
dx = 2.*r_earth/math.sqrt(3.)*math.sqrt(4.*math.pi/(num_cells))
dt = nu_cfl/c*dx

solver = GravityWaveSolver(W2, W3, Wb, dt, c, N, Omega, r_earth,
                           monitor=True, hybridization=True,
                           solver_type=solver_type)
solver.initialize(u0, p0, b0)

tmax = dt*100
t = 0
output = File("results/shell/gw3d.pvd")
u, p, b = solver._state.split()
output.write(u, p, b)
while t < tmax:
    t += dt
    solver.solve()
    u, p, b = solver._state.split()
    output.write(u, p, b)
