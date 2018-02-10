"""
Credit for demo setup: Gusto development team
"""
from gusto import *
import itertools
from firedrake import as_vector, SpatialCoordinate, PeriodicIntervalMesh, \
    ExtrudedMesh, exp, sin, Function, parameters
from firedrake.petsc import PETSc
from argparse import ArgumentParser
from hybridization import HybridizedCompressibleSolver
import numpy as np
import sys


# Given a delta, return appropriate dt
delta_dt = {62.5: 1.,
            125.: 2.,
            250.: 3.,
            500.: 4.,
            1000.: 5.,
            2000.: 6.}

PETSc.Log.begin()
parser = ArgumentParser(description="""Nonhydrostatic gravity wave test by Skamarock and Klemp (1994).""",
                        add_help=False)

parser.add_argument("--delta",
                    default=2000.0,
                    type=float,
                    choices=[2000.0, 1000.0, 500.0, 250.0, 125.0, 62.5],
                    help="Resolution for the simulation.")

parser.add_argument("--hybridization",
                    action="store_true",
                    help="Use a hybridized compressible solver.")

parser.add_argument("--test",
                    action="store_true",
                    help="Enable a quick test run.")

parser.add_argument("--profile",
                    action="store_true",
                    help="Turn on profiling for a 20 time-step run.")

parser.add_argument("--dumpfreq",
                    default=5,
                    type=int,
                    action="store",
                    help="Dump frequency of output files.")

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
    tmax = 20*delta_dt[args.delta]

if args.test:
    tmax = delta_dt[args.delta]

if not args.test and not args.profile:
    tmax = 3600.

delta = args.delta
dt = delta_dt[delta]
hybrid = bool(args.hybridization)
PETSc.Sys.Print("""
Problem parameters:\n
Test case: Skamarock and Klemp gravity wave.\n
Hybridized compressible solver: %s,\n
delta: %s,\n
Profiling: %s,\n
Test run: %s,\n
Dump frequency: %s.\n
""" % (hybrid, delta, bool(args.profile), bool(args.test), args.dumpfreq))

PETSc.Sys.Print("Initializing problem with dt: %s and tmax: %s.\n" % (dt,
                                                                      tmax))
H = 1.0e4  # Height position of the model top
L = 3.0e5

nlayers = int(2*H/delta)  # horizontal layers
columns = int(L/delta)    # number of columns
m = PeriodicIntervalMesh(columns, L)

# build volume mesh
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

points_x = np.linspace(0., L, 100)
points_z = [H/2.]
points = np.array([p for p in itertools.product(points_x, points_z)])

fieldlist = ['u', 'rho', 'theta']
timestepping = TimesteppingParameters(dt=dt)

if hybrid:
    dirname = 'hybrid_sk_nonlinear_dx%s_dt%s' % (delta, dt)
else:
    dirname = 'sk_nonlinear_dx%s_dt%s' % (delta, dt)

output = OutputParameters(dirname=dirname,
                          dumpfreq=args.dumpfreq,
                          dumplist=['u'],
                          perturbation_fields=['theta', 'rho'],
                          point_data=[('theta_perturbation', points)])
parameters = CompressibleParameters()
diagnostics = Diagnostics(*fieldlist)
diagnostic_fields = [CourantNumber()]

state = State(mesh, vertical_degree=1, horizontal_degree=1,
              family="CG",
              timestepping=timestepping,
              output=output,
              parameters=parameters,
              diagnostics=diagnostics,
              fieldlist=fieldlist,
              diagnostic_fields=diagnostic_fields)

# Initial conditions
u0 = state.fields("u")
rho0 = state.fields("rho")
theta0 = state.fields("theta")

# spaces
Vu = u0.function_space()
Vt = theta0.function_space()
Vr = rho0.function_space()

# Thermodynamic constants required for setting initial conditions
# and reference profiles
g = parameters.g
N = parameters.N
p_0 = parameters.p_0
c_p = parameters.cp
R_d = parameters.R_d
kappa = parameters.kappa

x, z = SpatialCoordinate(mesh)

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = 300.
thetab = Tsurf*exp(N**2*z/g)

theta_b = Function(Vt).interpolate(thetab)
rho_b = Function(Vr)

# Calculate hydrostatic Pi
compressible_hydrostatic_balance(state, theta_b, rho_b)

a = 5.0e3
deltaTheta = 1.0e-2
theta_pert = deltaTheta*sin(np.pi*z/H)/(1 + (x - L/2)**2/a**2)
theta0.interpolate(theta_b + theta_pert)
rho0.assign(rho_b)
u0.project(as_vector([20.0, 0.0]))

state.initialise([('u', u0),
                  ('rho', rho0),
                  ('theta', theta0)])
state.set_reference_profiles([('rho', rho_b),
                              ('theta', theta_b)])

# Set up advection schemes
ueqn = EulerPoincare(state, Vu)
rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")

supg = True
if supg:
    thetaeqn = SUPGAdvection(state, Vt,
                             supg_params={"dg_direction": "horizontal"},
                             equation_form="advective")
else:
    thetaeqn = EmbeddedDGAdvection(state, Vt, equation_form="advective")

advected_fields = []
advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
advected_fields.append(("rho", SSPRK3(state, rho0, rhoeqn)))
advected_fields.append(("theta", SSPRK3(state, theta0, thetaeqn)))

# Set up linear solver
if hybrid:
    linear_solver = HybridizedCompressibleSolver(state)
else:
    linear_solver = CompressibleSolver(state)

# Set up forcing
compressible_forcing = CompressibleForcing(state)

# Build time stepper
stepper = CrankNicolson(state, advected_fields, linear_solver,
                        compressible_forcing)

PETSc.Sys.Print("Starting simulation...\n")
stepper.run(t=0, tmax=tmax)
