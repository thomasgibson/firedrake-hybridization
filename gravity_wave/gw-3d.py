from firedrake import *
from function_spaces import construct_spaces
from solver import GravityWaveSolver
from gaussian import MultipleGaussianExpression
import math


# Spherical shell (Earth with layers)
# Number of vertical layers
nlayer = 6
# Radius of earth
r_earth = 6.371e6
# Thickness of spherical shell
thickness = 1.0e4

base = IcosahedralSphereMesh(r_earth,
                             refinement_level=5,
                             degree=3)
x = SpatialCoordinate(base)
base.init_cell_orientations(x)
mesh = ExtrudedMesh(base, extrusion_type='radial',
                    layers=nlayer, layer_height=thickness/nlayer)
num_cells = mesh.cell_set.size

g = MultipleGaussianExpression(16, r_earth, thickness)
expression = Expression(str(g))

W2, W3, Wb, _ = construct_spaces(mesh, order=1)

u0 = Function(W2).assign(0.0)
b0 = Function(Wb).assign(0.0)
p0 = Function(W3).interpolate(expression)

nu_cfl = 10.
dx = 2.*r_earth/math.sqrt(3.)*math.sqrt(4.*math.pi/(num_cells))
dt = nu_cfl/300.0*dx
c = 300
N = 0.01

solver = GravityWaveSolver(W2, W3, Wb, dt, c, N,
                           monitor=True, hybridization=True)
solver.initialize(u0, p0, b0)

tmax = dt*20
t = 0
output = File("results/shell/gw3d.pvd")
u, p, b = solver._state.split()
output.write(u, p, b)
while t < tmax:
    t += dt
    solver.solve()
    u, p, b = solver._state.split()
    output.write(u, p, b)
