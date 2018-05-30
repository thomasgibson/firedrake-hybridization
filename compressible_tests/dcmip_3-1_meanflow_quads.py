from gusto import *
from firedrake import CubedSphereMesh, ExtrudedMesh, Expression, \
    VectorFunctionSpace, FunctionSpace, Function, SpatialCoordinate, \
    as_vector
from firedrake import exp, acos, cos, sin, parameters
from firedrake.petsc import PETSc
from argparse import ArgumentParser
import numpy as np
import sys


parameters["pyop2_options"]["lazy_evaluation"] = False


PETSc.Log.begin()
parser = ArgumentParser(description=("""
DCMIP Test 3-1 Non-orographic gravity waves on a small planet.
"""), add_help=False)

parser.add_argument("--hybridization",
                    action="store_true",
                    help="Use a hybridized compressible solver.")

parser.add_argument("--test",
                    action="store_true",
                    help="Enable a quick test run.")

parser.add_argument("--profile",
                    action="store_true",
                    help="Turn on profiling.")

parser.add_argument("--dumpfreq",
                    default=100,
                    type=int,
                    action="store",
                    help="Dump frequency of output files.")

parser.add_argument("--dt",
                    default=10.,
                    type=float,
                    action="store",
                    help="Time step size (seconds)")

parser.add_argument("--tmax",
                    default=100.,
                    type=float,
                    action="store",
                    help="Max time (s). Max test time was set to 3600s.")

parser.add_argument("--refinements",
                    default=4,
                    type=int,
                    action="store",
                    help="Resolution scaling parameter.")

parser.add_argument("--layers",
                    default=16,
                    type=int,
                    action="store",
                    help="Number of vertical layers.")

parser.add_argument("--debug",
                    action="store_true",
                    help="Turn on KSP monitors")

parser.add_argument("--rtol",
                    default=1.0e-6,
                    type=float,
                    help="Rtolerance for the linear solve.")

parser.add_argument("--help",
                    action="store_true",
                    help="Show help.")


args, _ = parser.parse_known_args()

if args.help:
    help = parser.format_help()
    PETSc.Sys.Print("%s\n" % help)
    sys.exit(1)

dt = args.dt                    # Time-step size (s)
tmax = args.tmax                # Maximum time (s)
nlayers = args.layers           # Number of vertical layers
refinements = args.refinements  # Number of horiz. cells = 20*(4^refinements)

if args.profile:
    tmax = 5*dt

if args.test:
    tmax = dt

hybrid = bool(args.hybridization)
PETSc.Sys.Print("""
Problem parameters:\n
Test case DCMIP 3-1: Non-orographic gravity waves on a small planet.\n
Hybridized compressible solver: %s,\n
Time-step size: %s,\n
Horizontal refinements: %s,\n
Vertical layers: %s,\n
Profiling: %s,\n
Max time: %s,\n
Dump frequency: %s.\n
""" % (hybrid, dt, refinements, nlayers,
       bool(args.profile), args.tmax, args.dumpfreq))

PETSc.Sys.Print("Initializing problem with dt: %s and tmax: %s.\n" % (dt,
                                                                      tmax))

# Set up problem parameters
parameters = CompressibleParameters()
a_ref = 6.37122e6               # Radius of the Earth (m)
X = 125.0                       # Reduced-size Earth reduction factor
a = a_ref/X                     # Scaled radius of planet (m)
g = parameters.g                # Acceleration due to gravity (m/s^2)
N = parameters.N                # Brunt-Vaisala frequency (1/s)
p_0 = parameters.p_0            # Reference pressure (Pa, not hPa)
c_p = parameters.cp             # SHC of dry air at constant pressure (J/kg/K)
R_d = parameters.R_d            # Gas constant for dry air (J/kg/K)
kappa = parameters.kappa        # R_d/c_p
T_eq = 300.0                    # Isothermal atmospheric temperature (K)
p_eq = 1000.0 * 100.0           # Reference surface pressure at the equator
u_0 = 20.0                      # Maximum amplitude of the zonal wind (m/s)
d = 5000.0                      # Width parameter for Theta'
lamda_c = 2.0*np.pi/3.0         # Longitudinal centerpoint of Theta'
phi_c = 0.0                     # Latitudinal centerpoint of Theta' (equator)
deltaTheta = 1.0                # Maximum amplitude of Theta' (K)
L_z = 20000.0                   # Vertical wave length of the Theta' perturb.

# Cubed-sphere mesh
m = CubedSphereMesh(radius=a,
                    refinement_level=refinements,
                    degree=2)

# Build volume mesh
z_top = 1.0e4            # Height position of the model top
mesh = ExtrudedMesh(m, layers=nlayers,
                    layer_height=z_top/nlayers,
                    extrusion_type="radial")

# Space for initialising velocity (using this ensures things are in layers)
W_VectorCG1 = VectorFunctionSpace(mesh, "CG", 1)
W_CG1 = FunctionSpace(mesh, "CG", 1)

# Create polar coordinates:
# Since we use a CG1 field, this is constant on layers
z_expr = Expression("sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]) - a", a=a)
z = Function(W_CG1).interpolate(z_expr)
lat_expr = Expression("asin(x[2]/sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]))")
lat = Function(W_CG1).interpolate(lat_expr)
lon = Function(W_CG1).interpolate(Expression("atan2(x[1], x[0])"))

fieldlist = ['u', 'rho', 'theta']
timestepping = TimesteppingParameters(dt=dt, maxk=4, maxi=1)

dirname = 'meanflow_ref'
if hybrid:
    dirname += '_hybridization'

output = OutputParameters(dumpfreq=args.dumpfreq, dirname=dirname,
                          perturbation_fields=['theta', 'rho'])
diagnostics = Diagnostics(*fieldlist)

state = State(mesh, vertical_degree=1, horizontal_degree=1,
              family="RTCF",
              timestepping=timestepping,
              output=output,
              parameters=parameters,
              diagnostics=diagnostics,
              fieldlist=fieldlist)

# Initial conditions
u0 = state.fields.u
theta0 = state.fields.theta
rho0 = state.fields.rho

# spaces
Vu = u0.function_space()
Vt = theta0.function_space()
Vr = rho0.function_space()

# Initial conditions with u0
x = SpatialCoordinate(mesh)
u_max = 20.
uexpr = as_vector([-u_max*x[1]/a, u_max*x[0]/a, 0.0])
u0.project(uexpr)

# Surface temperature
G = g**2/(N**2*c_p)
Ts_expr = G + (T_eq-G)*exp(-(u_max*N**2/(4*g*g))*u_max*(cos(2.0*lat)-1.0))
Ts = Function(W_CG1).interpolate(Ts_expr)

# Surface pressure
ps_expr = p_eq*exp((u_max/(4.0*G*R_d))*u_max*(cos(2.0*lat)-1.0))*(Ts/T_eq)**(1.0/kappa)
ps = Function(W_CG1).interpolate(ps_expr)

# Background pressure
p_expr = ps*(1 + G/Ts*(exp(-N**2*z/g)-1))**(1.0/kappa)
p = Function(W_CG1).interpolate(p_expr)

# Background temperature
Tb_expr = G*(1 - exp(N**2*z/g)) + Ts*exp(N**2*z/g)
Tb = Function(W_CG1).interpolate(Tb_expr)

# Background potential temperature
thetab_expr = Tb*(p_0/p)**kappa
thetab = Function(W_CG1).interpolate(thetab_expr)
theta_b = Function(theta0.function_space()).interpolate(thetab)
rho_b = Function(rho0.function_space())
sin_tmp = sin(lat) * sin(phi_c)
cos_tmp = cos(lat) * cos(phi_c)
r = a*acos(sin_tmp + cos_tmp*cos(lon-lamda_c))
s = (d**2)/(d**2 + r**2)
theta_pert = deltaTheta*s*sin(2*np.pi*z/L_z)
theta0.interpolate(theta_b)

# Compute the balanced density
PETSc.Sys.Print("Computing balanced density field...\n")
compressible_hydrostatic_balance(state,
                                 theta_b,
                                 rho_b,
                                 top=False,
                                 pi_boundary=(p/p_0)**kappa)
theta0.interpolate(theta_pert)
theta0 += theta_b
rho0.assign(rho_b)

state.initialise([('u', u0), ('rho', rho0), ('theta', theta0)])
state.set_reference_profiles([('rho', rho_b), ('theta', theta_b)])

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
    PETSc.Sys.Print("""
    Setting up hybridized solver with BCGS + GAMG on the traces.""")

    mg_params = {'ksp_type': 'richardson',
                 'ksp_max_it': 3,
                 'pc_type': 'bjacobi',
                 'sub_pc_type': 'ilu'}

    solver_parameters = {'ksp_type': 'bcgs',
                         'ksp_rtol': args.rtol,
                         'pc_type': 'gamg',
                         'pc_gamg_sym_graph': True,
                         'pc_gamg_reuse_interpolation': True,
                         'mg_levels': mg_params}
    if args.debug:
        solver_parameters['ksp_monitor_true_residual'] = True

    PETSc.Sys.Print("""
    Full solver options:\n
    %s
    """ % solver_parameters)
    linear_solver = HybridizedCompressibleSolver(state,
                                                 solver_parameters=solver_parameters,
                                                 overwrite_solver_parameters=True)
else:
    PETSc.Sys.Print("""
    Setting up GMRES fieldsplit solver with Schur complement PC.""")

    # Aggressive AMG procedure
    mg_params = {'ksp_type': 'chebyshev',
                 'ksp_chebyshev_esteig': True,
                 'ksp_max_it': 2,
                 'pc_type': 'bjacobi',
                 'sub_pc_type': 'ilu'}

    solver_parameters = {'pc_type': 'fieldsplit',
                         'pc_fieldsplit_type': 'schur',
                         'ksp_type': 'gmres',
                         'ksp_rtol': args.rtol,
                         'ksp_max_it': 100,
                         'ksp_gmres_restart': 50,
                         'pc_fieldsplit_schur_fact_type': 'FULL',
                         'pc_fieldsplit_schur_precondition': 'selfp',
                         'fieldsplit_0': {'ksp_type': 'preonly',
                                          'pc_type': 'bjacobi',
                                          'sub_pc_type': 'ilu'},
                         'fieldsplit_1': {'ksp_type': 'preonly',
                                          'pc_type': 'gamg',
                                          'pc_gamg_sym_graph': True,
                                          'pc_gamg_reuse_interpolation': True,
                                          'mg_levels': mg_params}}
    if args.debug:
        solver_parameters['ksp_monitor_true_residual'] = True

    PETSc.Sys.Print("""
    Full solver options:\n
    %s
    """ % solver_parameters)
    linear_solver = CompressibleSolver(state,
                                       solver_parameters=solver_parameters,
                                       overwrite_solver_parameters=True)

# Set up forcing
compressible_forcing = CompressibleForcing(state)

# Build time stepper
stepper = CrankNicolson(state, advected_fields, linear_solver,
                        compressible_forcing)

PETSc.Sys.Print("Starting simulation...\n")
stepper.run(t=0, tmax=tmax)
