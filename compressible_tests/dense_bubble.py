"""
Credit for demo setup: Gusto development team
"""
from gusto import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, \
    SpatialCoordinate, Constant, DirichletBC, pi, cos, Function, sqrt, \
    conditional, parameters
from firedrake.petsc import PETSc
from argparse import ArgumentParser
from hybridization import HybridizedCompressibleSolver
import sys


# Given a delta, return appropriate dt
delta_dt = {50.: 0.25,
            100.: 0.5,
            200.: 1.,
            400.: 2.,
            800.: 4.}

PETSc.Log.begin()
parser = ArgumentParser(description="""
Dense bubble test by Straka et al (1993).
""", add_help=False)

parser.add_argument("--delta",
                    default=800.0,
                    type=float,
                    choices=[800.0, 400.0, 200.0, 100.0, 50.0],
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

if args.profile:
    # Ensures accurate timing of parallel loops
    parameters["pyop2_options"]["lazy_evaluation"] = False
    tmax = 20*delta_dt[args.delta]

if args.test:
    tmax = delta_dt[args.delta]

if not args.test and not args.profile:
    tmax = 15.*60.

delta = args.delta
dt = delta_dt[delta]
hybrid = bool(args.hybridization)
PETSc.Sys.Print("""
Problem parameters:\n
Test case: Straka falling dense bubble.\n
Hybridized compressible solver: %s,\n
delta: %s,\n
Profiling: %s,\n
Test run: %s,\n
Dump frequency: %s.\n
""" % (hybrid, delta, bool(args.profile), bool(args.test), args.dumpfreq))

PETSc.Sys.Print("Initializing problem with dt: %s and tmax: %s.\n" % (dt,
                                                                      tmax))
L = 51200.
H = 6400.  # Height position of the model top

if hybrid:
    dirname = "hybrid_db_dx%s_dt%s" % (delta, dt)
else:
    dirname = "db_dx%s_dt%s" % (delta, dt)

nlayers = int(H/delta)  # horizontal layers
columns = int(L/delta)  # number of columns

m = PeriodicIntervalMesh(columns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

fieldlist = ['u', 'rho', 'theta']
timestepping = TimesteppingParameters(dt=dt, maxk=4, maxi=1)
output = OutputParameters(dirname=dirname,
                          dumpfreq=args.dumpfreq,
                          dumplist=['u'],
                          perturbation_fields=['theta', 'rho'])
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

# Isentropic background state
Tsurf = Constant(300.)

theta_b = Function(Vt).interpolate(Tsurf)
rho_b = Function(Vr)

# Calculate hydrostatic Pi
params = {'pc_type': 'fieldsplit',
          'pc_fieldsplit_type': 'schur',
          'ksp_type': 'gmres',
          'ksp_monitor_true_residual': True,
          'ksp_max_it': 1000,
          'ksp_gmres_restart': 50,
          'pc_fieldsplit_schur_fact_type': 'FULL',
          'pc_fieldsplit_schur_precondition': 'selfp',
          'fieldsplit_0': {'ksp_type': 'preonly',
                           'pc_type': 'bjacobi',
                           'sub_pc_type': 'ilu'},
          'fieldsplit_1': {'ksp_type': 'preonly',
                           'pc_type': 'gamg',
                           'pc_gamg_sym_graph': True,
                           'mg_levels': {'ksp_type': 'chebyshev',
                                         'ksp_chebyshev_esteig': True,
                                         'ksp_max_it': 5,
                                         'pc_type': 'bjacobi',
                                         'sub_pc_type': 'ilu'}}}

PETSc.Sys.Print("Computing hydrostatic Exner pressure...\n")
compressible_hydrostatic_balance(state,
                                 theta_b,
                                 rho_b,
                                 solve_for_rho=True,
                                 params=params)

x = SpatialCoordinate(mesh)
a = 5.0e3
deltaTheta = 1.0e-2
xc = 0.5*L
xr = 4000.
zc = 3000.
zr = 2000.
r = sqrt(((x[0]-xc)/xr)**2 + ((x[1]-zc)/zr)**2)
theta_pert = conditional(r > 1., 0., -7.5*(1.+cos(pi*r)))
theta0.interpolate(theta_b + theta_pert)
rho0.assign(rho_b)

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
    thetaeqn = EmbeddedDGAdvection(state, Vt,
                                   equation_form="advective")

advected_fields = []
advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
advected_fields.append(("rho", SSPRK3(state, rho0, rhoeqn)))
advected_fields.append(("theta", SSPRK3(state, theta0, thetaeqn)))

# Set up linear solver
if hybrid:
    if args.debug:
        solver_parameters = {'ksp_type': 'bcgs',
                             'pc_type': 'gamg',
                             'ksp_rtol': 1.0e-8,
                             'ksp_monitor_true_residual': True,
                             'mg_levels': {'ksp_type': 'richardson',
                                           'ksp_max_it': 3,
                                           'pc_type': 'bjacobi',
                                           'sub_pc_type': 'ilu'}}
        linear_solver = HybridizedCompressibleSolver(state, solver_parameters=solver_parameters,
                                                     overwrite_solver_parameters=True)
    else:
        linear_solver = HybridizedCompressibleSolver(state)
else:
    if args.debug:
        solver_parameters = {'pc_type': 'fieldsplit',
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
                                                            'sub_pc_type': 'ilu'}}}
        linear_solver = CompressibleSolver(state, solver_parameters=solver_parameters,
                                           overwrite_solver_parameters=True)
    else:
        linear_solver = CompressibleSolver(state)

# Set up forcing
compressible_forcing = CompressibleForcing(state)

bcs = [DirichletBC(Vu, 0.0, "bottom"),
       DirichletBC(Vu, 0.0, "top")]
diffused_fields = [("u", InteriorPenalty(state, Vu, kappa=75.,
                                         mu=Constant(10./delta), bcs=bcs)),
                   ("theta", InteriorPenalty(state, Vt, kappa=75.,
                                             mu=Constant(10./delta)))]

# Build time stepper
stepper = CrankNicolson(state, advected_fields, linear_solver,
                        compressible_forcing, diffused_fields)

PETSc.Sys.Print("Starting simulation...\n")
stepper.run(t=0, tmax=tmax)
