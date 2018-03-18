from firedrake import *
from firedrake.petsc import PETSc
from function_spaces import construct_spaces
from ksp_monitor import KSPMonitor
from solver import GravityWaveSolver
from argparse import ArgumentParser
import math
import numpy as np
import sys


PETSc.Log.begin()
parser = ArgumentParser(description="""Linear gravity wave system.""",
                        add_help=False)

parser.add_argument("--refinements",
                    default=4,
                    type=int,
                    help=("Number of refinements when generating the "
                          "spherical base mesh."))

parser.add_argument("--extrusion_levels",
                    default=20,
                    type=int,
                    help="Number of vertical levels in the extruded mesh.")

parser.add_argument("--hybridization",
                    action="store_true",
                    help=("Use a hybridized mixed method to solve the "
                          "gravity wave equations."))

parser.add_argument("--hybrid_local_solve_type",
                    default=None,
                    choices=["colPivHouseholderQr", "partialPivLu",
                             "fullPivLu", "householderQr", "jacobiSvd",
                             "fullPivHouseholderQr", "llt", "ldlt"],
                    help=("Solver type for local solves when "
                          "using hybridization."))

parser.add_argument("--hybrid_invert_type",
                    default=None,
                    choices=["colPivHouseholderQr", "partialPivLu",
                             "fullPivLu", "householderQr", "jacobiSvd",
                             "fullPivHouseholderQr", "llt", "ldlt"],
                    help="Use a factorization to invert local Slate tensors?")

parser.add_argument("--solver_type",
                    default="gamg",
                    choices=["hypre", "gamg", "preonly-gamg",
                             "direct", "hybrid_mg"],
                    help="Solver type for the linear solver.")

parser.add_argument("--nu_cfl",
                    default=1,
                    type=int,
                    help="Value for the horizontal courant number.")

parser.add_argument("--num_timesteps",
                    default=1,
                    type=int,
                    help="Number of time steps to take.")

parser.add_argument("--order",
                    default=1,
                    type=int,
                    help="Order of the compatible mixed method.")

parser.add_argument("--rtol",
                    default=1.0E-6,
                    type=float,
                    help="Rtolerance for the linear solve.")

parser.add_argument("--test",
                    action="store_true",
                    help="Enable a quick test run with ksp monitors.")

parser.add_argument("--profile",
                    action="store_true",
                    help="Turn on profiler for simulation timings.")

parser.add_argument("--output",
                    action="store_true",
                    help="Write output.")

parser.add_argument("--write_data",
                    action="store_true",
                    help="Write data file")

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

# Save this for later...
if bool(None):
    n_mglvls = reflvl - 3
    base_h = MeshHierarchy(IcosahedralSphereMesh(r_earth,
                                                 refinement_level=3,
                                                 degree=3),
                           n_mglvls)
    mh = ExtrudedMeshHierarchy(base_h, extrusion_type='radial',
                               layers=nlayer, layer_height=thickness/nlayer)
    mesh = mh[-1]
else:
    base = IcosahedralSphereMesh(r_earth,
                                 refinement_level=reflvl,
                                 degree=3)

    # Extruded mesh
    mesh = ExtrudedMesh(base, extrusion_type='radial',
                        layers=nlayer, layer_height=thickness/nlayer)

# Compatible finite element spaces
W2, W3, Wb, _ = construct_spaces(mesh, order=order)

# Physical constants and time-stepping parameters
c = 343
N = 0.01
Omega = 7.292e-5
nu_cfl = args.nu_cfl
num_cells = mesh.cell_set.size
dx = 2.*r_earth/math.sqrt(3.)*math.sqrt(4.*math.pi/(num_cells))
dt = nu_cfl/c*dx

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
p_eq = 1000.0 * 100.0
g = 9.810616
R_d = 287.0
T_eq = 300.0
c_p = 1004.5
kappa = 2.0/7.0
G = g**2/(N**2*c_p)
Ts = Function(W_CG1).interpolate(G + (T_eq-G)*exp(
    -(u_max*N**2/(4*g*g))*u_max*(cos(2.0*lat)-1.0)))
psexp = p_eq*exp((u_max/(4.0*G*R_d))*u_max*(cos(2.0*lat)-1.0))*(Ts/T_eq)**(1.0/kappa)
p0.interpolate(psexp)

# Setup linear solvers
solver = GravityWaveSolver(W2, W3, Wb, dt, c, N, Omega, r_earth,
                           monitor=args.test, rtol=args.rtol,
                           hybridization=args.hybridization,
                           solver_type=args.solver_type,
                           local_invert_method=args.hybrid_invert_type,
                           local_solve_method=args.hybrid_local_solve_type)

# Initialize
solver.initialize(u0, p0, b0)

if args.test:
    tmax = dt
else:
    tmax = dt*args.num_timesteps

# RUN!
t = 0
u, p, b = solver._state.split()

if not args.profile:
    solver.ksp_monitor = KSPMonitor()

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

if args.write_data:
    solver.ksp_monitor.write_to_csv()
    print(solver.up_residual_reductions)
