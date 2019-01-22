from gusto import *
import itertools
from firedrake import as_vector, SpatialCoordinate, PeriodicIntervalMesh, \
    ExtrudedMesh, exp, sin, Function, parameters, FunctionSpace, \
    VectorFunctionSpace, BrokenElement
from firedrake.petsc import PETSc
from argparse import ArgumentParser
import numpy as np
import sys


PETSc.Log.begin()
parser = ArgumentParser(description=("""
Nonhydrostatic gravity wave test by Skamarock and Klemp (1994).
"""), add_help=False)

parser.add_argument("--hybridization",
                    action="store_true",
                    help="Use a hybridized compressible solver.")

parser.add_argument("--test",
                    action="store_true",
                    help="Enable a quick test run.")

parser.add_argument("--profile",
                    action="store_true",
                    help="Turn on profiling for a 20 time-step run.")

parser.add_argument("--dt",
                    action="store",
                    default=10.0,
                    type=float,
                    help="Time step size (s)")

parser.add_argument("--recovered",
                    action="store_true",
                    help="Use recovered spaces advection scheme.")

parser.add_argument("--dumpfreq",
                    default=5,
                    type=int,
                    action="store",
                    help="Dump frequency of output files.")

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

hybrid = bool(args.hybridization)

H = 1.0e4  # Height position of the model top
L = 3.0e5

nlayers = 10         # horizontal layers
columns = 300        # number of columns
dt = args.dt         # Time steps (s)

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
Hybridized compressible solver: %s,\n
Time-step size: %s,\n
Profiling: %s,\n
Test run: %s,\n
Dx (m): %s,\n
Dz (m): %s,\n
CFL: %s,\n
Dump frequency: %s.\n
""" % (hybrid, dt, bool(args.profile), bool(args.test),
       dx, dz, cfl, args.dumpfreq))

if args.profile:
    # Ensures accurate timing of parallel loops
    parameters["pyop2_options"]["lazy_evaluation"] = False
    tmax = 20*dt

if args.test:
    tmax = dt

if not args.test and not args.profile:
    tmax = 3600.

PETSc.Sys.Print("Initializing problem with dt: %s and tmax: %s.\n" % (dt,
                                                                      tmax))

# build volume mesh
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

fieldlist = ['u', 'rho', 'theta']
timestepping = TimesteppingParameters(dt=dt)

if hybrid:
    dirname = 'hybrid_sk_nonlinear_dx%s_dz%s_dt%s' % (dx, dz, dt)
else:
    dirname = 'sk_nonlinear_dx%s_dz%s_dt%s' % (dx, dz, dt)

points_x = np.linspace(0., L, 100)
points_z = [H/2.]
points = np.array([p for p in itertools.product(points_x, points_z)])

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
advected_fields = []
recovered = args.recovered
if recovered:
    VDG1 = FunctionSpace(mesh, "DG", 1)
    VCG1 = FunctionSpace(mesh, "CG", 1)
    Vt_brok = FunctionSpace(mesh, BrokenElement(Vt.ufl_element()))
    Vu_DG1 = VectorFunctionSpace(mesh, "DG", 1)
    Vu_CG1 = VectorFunctionSpace(mesh, "CG", 1)

    u_spaces = (Vu_DG1, Vu_CG1, Vu)
    rho_spaces = (VDG1, VCG1, Vr)
    theta_spaces = (VDG1, VCG1, Vt_brok)

    ueqn = EmbeddedDGAdvection(state, Vu, equation_form="advective",
                               recovered_spaces=u_spaces)
    rhoeqn = EmbeddedDGAdvection(state, Vr, equation_form="continuity",
                                 recovered_spaces=rho_spaces)
    thetaeqn = EmbeddedDGAdvection(state, Vt, equation_form="advective",
                                   recovered_spaces=theta_spaces)
    advected_fields.append(('u', SSPRK3(state, u0, ueqn)))
else:
    ueqn = EulerPoincare(state, Vu)
    rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")

    supg = True
    if supg:
        thetaeqn = SUPGAdvection(state, Vt,
                                 supg_params={"dg_direction": "horizontal"},
                                 equation_form="advective")
    else:
        thetaeqn = EmbeddedDGAdvection(state, Vt, equation_form="advective")

    advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))

advected_fields.append(("rho", SSPRK3(state, rho0, rhoeqn)))
advected_fields.append(("theta", SSPRK3(state, theta0, thetaeqn)))

# Set up linear solver
if hybrid:
    if args.debug:
        inner_parameters = {
             'ksp_type': 'fgmres',
             'ksp_rtol': 1.0e-8,
             'ksp_atol': 1.0e-8,
             'ksp_max_it': 100,
             'pc_type': 'gamg',
             'pc_gamg_sym_graph': True,
             'mg_levels': {'ksp_type': 'gmres',
                           'ksp_max_its': 5,
                           'pc_type': 'bjacobi',
                           'sub_pc_type': 'ilu'}
         }
        inner_parameters['ksp_monitor_true_residual'] = True

        # Use Firedrake static condensation interface
        solver_parameters = {
            'mat_type': 'matfree',
            'pmat_type': 'matfree',
            'ksp_type': 'preonly',
            'pc_type': 'python',
            'pc_python_type': 'firedrake.SCPC',
            'pc_sc_eliminate_fields': '0, 1',
            'condensed_field': inner_parameters
        }
        linear_solver = HybridizedCompressibleSolver(state, solver_parameters=solver_parameters,
                                                     overwrite_solver_parameters=True)
    else:
        linear_solver = HybridizedCompressibleSolver(state)
else:
    if args.debug:
        solver_parameters = {
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'schur',
            'ksp_type': 'gmres',
            'ksp_monitor_true_residual': True,
            'ksp_max_it': 100,
            'ksp_gmres_restart': 50,
            'pc_fieldsplit_schur_fact_type': 'FULL',
            'pc_fieldsplit_schur_precondition': 'selfp',
            'fieldsplit_0': {'ksp_type': 'preonly',
                             'pc_type': 'bjacobi',
                             'sub_pc_type': 'ilu'},
            'fieldsplit_1': {'ksp_type': 'preonly',
                             'pc_type': 'gamg',
                             'mg_levels': {'ksp_type': 'chebyshev',
                                           'ksp_chebyshev_esteig': True,
                                           'ksp_max_it': 1,
                                           'pc_type': 'bjacobi',
                                           'sub_pc_type': 'ilu'}}
        }
        linear_solver = CompressibleSolver(state, solver_parameters=solver_parameters,
                                           overwrite_solver_parameters=True)
    else:
        linear_solver = CompressibleSolver(state)

# Set up forcing
if recovered:
    compressible_forcing = CompressibleForcing(state, euler_poincare=False)
else:
    compressible_forcing = CompressibleForcing(state)

# Build time stepper
stepper = CrankNicolson(state, advected_fields, linear_solver,
                        compressible_forcing)

PETSc.Sys.Print("Starting simulation...\n")
stepper.run(t=0, tmax=tmax)
