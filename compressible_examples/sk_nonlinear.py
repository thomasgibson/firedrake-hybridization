from gusto import *
import itertools
from firedrake import (as_vector, SpatialCoordinate,
                       PeriodicIntervalMesh,
                       ExtrudedMesh, exp, sin, Function,
                       FunctionSpace, VectorFunctionSpace,
                       BrokenElement)
from firedrake.petsc import PETSc
from argparse import ArgumentParser
import numpy as np
import sys


PETSc.Log.begin()

parser = ArgumentParser(description=("""
Nonhydrostatic gravity wave test based on that of Skamarock and Klemp (1994).
"""), add_help=False)

parser.add_argument("--test",
                    action="store_true",
                    help="Enable a quick test run.")

parser.add_argument("--dt",
                    action="store",
                    default=6.0,
                    type=float,
                    help="Time step size (s)")

parser.add_argument("--res",
                    default=1,
                    type=int,
                    action="store",
                    help="Resolution scaling parameter")

parser.add_argument("--debug",
                    action="store_true",
                    help="Turn on KSP monitors")

parser.add_argument("--help",
                    action="store_true",
                    help="Show help.")

args, _ = parser.parse_known_args()

if args.help:
    help = parser.format_help()
    PETSc.Sys.Print("%s\n" % help)
    sys.exit(1)

res = args.res
nlayers = res*10         # horizontal layers
columns = res*300        # number of columns
dt = args.dt             # Time steps (s)

if args.test:
    tmax = dt
else:
    tmax = 3600.

H = 1.0e4  # Height position of the model top
L = 3.0e5

PETSc.Sys.Print("""
Number of vertical layers: %s,\n
Number of horizontal columns: %s.\n
""" % (nlayers, columns))

m = PeriodicIntervalMesh(columns, L)

dx = L / columns
cfl = 20.0 * dt / dx
dz = H / nlayers

PETSc.Sys.Print("""
Problem parameters:\n
Test case: Skamarock and Klemp gravity wave.\n
Time-step size: %s,\n
Test run: %s,\n
Dx (m): %s,\n
Dz (m): %s,\n
CFL: %s\n
""" % (dt,
       bool(args.test),
       dx,
       dz,
       cfl))

PETSc.Sys.Print("Initializing problem with dt: %s and tmax: %s.\n" % (dt,
                                                                      tmax))

# build volume mesh
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

fieldlist = ['u', 'rho', 'theta']
timestepping = TimesteppingParameters(dt=dt)

dirname = 'sk_nonlinear_dx%s_dz%s_dt%s' % (dx, dz, dt)

points_x = np.linspace(0., L, 100)
points_z = [H/2.]
points = np.array([p for p in itertools.product(points_x, points_z)])

dumptime = 100  # print every 100s
dumpfreq = int(dumptime / dt)
PETSc.Sys.Print("Output frequency: %s\n" % dumpfreq)

output = OutputParameters(dirname=dirname,
                          dumpfreq=dumpfreq,
                          dumplist=['u'],
                          perturbation_fields=['theta', 'rho'],
                          point_data=[('theta_perturbation', points)],
                          log_level='INFO')

parameters = CompressibleParameters()
diagnostics = Diagnostics(*fieldlist)
diagnostic_fields = [CourantNumber()]

state = State(mesh,
              vertical_degree=1,
              horizontal_degree=1,
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
PETSc.Sys.Print("Computing hydrostatic varaibles...\n")

# Use vertical hybridization preconditioner for the balance initialization
piparams = {'ksp_type': 'preonly',
            'pc_type': 'python',
            'mat_type': 'matfree',
            'pc_python_type': 'gusto.VerticalHybridizationPC',
            # Vertical trace system is only coupled vertically in columns
            # block ILU is a direct solver!
            'vert_hybridization': {'ksp_type': 'preonly',
                                   'pc_type': 'bjacobi',
                                   'sub_pc_type': 'ilu'}}

compressible_hydrostatic_balance(state,
                                 theta_b,
                                 rho_b,
                                 params=piparams)

PETSc.Sys.Print("Finished computing hydrostatic varaibles...\n")

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
    thetaeqn = SUPGAdvection(state, Vt, equation_form="advective")
else:
    thetaeqn = EmbeddedDGAdvection(state, Vt, equation_form="advective",
                                   options=EmbeddedDGOptions())

advected_fields = []
advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
advected_fields.append(("rho", SSPRK3(state, rho0, rhoeqn)))
advected_fields.append(("theta", SSPRK3(state, theta0, thetaeqn)))

# Set up linear solver
solver_parameters = {'mat_type': 'matfree',
                     'ksp_type': 'preonly',
                     'pc_type': 'python',
                     'pc_python_type': 'firedrake.SCPC',
                     'pc_sc_eliminate_fields': '0, 1',
                     # The reduced operator is not symmetric
                     'condensed_field': {'ksp_type': 'fgmres',
                                         'ksp_rtol': 1.0e-8,
                                         'ksp_atol': 1.0e-8,
                                         'ksp_max_it': 100,
                                         'pc_type': 'gamg',
                                         'pc_gamg_sym_graph': None,
                                         'mg_levels': {'ksp_type': 'gmres',
                                                       'ksp_max_it': 5,
                                                       'pc_type': 'bjacobi',
                                                       'sub_pc_type': 'ilu'}}}
if args.debug:
    solver_parameters['condensed_field']['ksp_monitor_true_residual'] = None

linear_solver = CompressibleSolver(state,
                                   solver_parameters=solver_parameters,
                                   overwrite_solver_parameters=True)

# Set up forcing
compressible_forcing = CompressibleForcing(state)

# Build time stepper
stepper = CrankNicolson(state,
                        advected_fields,
                        linear_solver,
                        compressible_forcing)

PETSc.Sys.Print("Starting simulation...\n")
stepper.run(t=0, tmax=tmax)
