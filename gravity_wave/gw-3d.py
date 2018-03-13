from firedrake import *
from firedrake.petsc import PETSc
from function_spaces import construct_spaces
from solver import GravityWaveSolver
from argparse import ArgumentParser
import math
import numpy as np
import sys


PETSc.Log.begin()
parser = ArgumentParser(description="""Linear gravity wave system.""",
                        add_help=False)

parser.add_argument("--refinements",
                    default=3,
                    type=int,
                    choices=[3, 4, 5, 6, 7],
                    help=("Number of refinements when generating the "
                          "spherical base mesh."))

parser.add_argument("--extrusion_levels",
                    default=4,
                    type=int,
                    help="Number of vertical levels in the extruded mesh.")

parse.add_argument("--cubedsphere",
                   action="store_true",
                   help="Use an extruded cubed-sphere mesh.")

parser.add_argument("--hybridization",
                    action="store_true",
                    help=("Use a hybridized mixed method to solve the "
                          "gravity wave equations."))

parser.add_argument("--order",
                    default=1,
                    type=int,
                    help="Order of the compatible mixed method.")

parser.add_argument("--test",
                    action="store_true",
                    help="Enable a quick test run.")

parser.add_argument("--profile",
                    action="store_true",
                    help="Turn on profiler for simulation timings.")

parser.add_argument("--long_run",
                    action="store_true",
                    help="Run simulation for 100 time-steps")

parser.add_argument("--output",
                    action="store_true",
                    help="Write output.")

parser.add_argument("--help",
                    action="store_true",
                    help="Show help.")

args, _ = parser.parse_known_args()

if args.help:
    help = parser.format_help()
    PETSc.Sys.Print("%s\n" % help)
    sys.exit(1)

if args.profile:
    # Ensures accurate timing of parallel loops
    parameters["pyop2_options"]["lazy_evaluation"] = False

# Number of vertical layers
nlayer = args.extrusion_levels

# Radius of earth (scaled)
r_earth = 6.371e6/125.0

# Thickness of spherical shell
thickness = 1.0e4

order = args.order
reflvl = args.refinements

if args.cubedsphere:
    base = CubedSphereMesh(r_earth,
                           refinement_level=reflvl,
                           degree=3)
else:
    base = IcosahedralSphereMesh(r_earth,
                                 refinement_level=reflvl,
                                 degree=3)

# Extruded mesh
mesh = ExtrudedMesh(base, extrusion_type='radial',
                    layers=nlayer, layer_height=thickness/nlayer)

# Compatible finite element spaces
W2, W3, Wb, _ = construct_spaces(mesh, order=order)

# Initial condition for velocity
u0 = Function(W2)
x = SpatialCoordinate(mesh)
u_max = 20.
uexpr = as_vector([-u_max*x[1]/r_earth,
                   u_max*x[0]/r_earth, 0.0])
u0.project(uexpr)

# Initial condition for buoyancy
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

# Initial condition for pressure
p0 = Function(W3).assign(0.0)

# Physical constants and time-stepping parameters
c = 300
N = 0.01
Omega = 7.292e-5
nu_cfl = 8.
num_cells = mesh.cell_set.size
dx = 2.*r_earth/math.sqrt(3.)*math.sqrt(4.*math.pi/(num_cells))
dt = nu_cfl/c*dx

# Setup linear solvers
solver = GravityWaveSolver(W2, W3, Wb, dt, c, N, Omega, r_earth,
                           monitor=args.test,
                           hybridization=args.hybridization)

# Initialize
solver.initialize(u0, p0, b0)

if args.test:
    tmax = 4*dt
elif args.long_run:
    tmax = dt*100
else:
    tmax = dt*20

# RUN!
t = 0
u, p, b = solver._state.split()

if args.output:
    output = File("results/gw3d/gw3d.pvd")
    output.write(u, p, b)

    while t < tmax:
        t += dt
        solver.solve()
        u, p, b = solver._state.split()
        output.write(u, p, b)
else:
    while t < tmax:
        t += dt
        solver.solve()
