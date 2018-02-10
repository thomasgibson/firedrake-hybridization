"""
Credit for demo setup: Gusto development team
"""
from gusto import *
from firedrake import FunctionSpace, as_vector, \
    VectorFunctionSpace, PeriodicIntervalMesh, ExtrudedMesh, \
    SpatialCoordinate, exp, pi, cos, Function, conditional, Mesh, sin, \
    op2, parameters
from firedrake.petsc import PETSc
from argparse import ArgumentParser
from hybridization import HybridizedCompressibleSolver
import sys


# Given a delta, return appropriate dt
delta_dt = {62.5: 1.,
            125.: 2.,
            250.: 3.,
            500.: 4.,
            1000.: 5.,
            2000.: 6.}

PETSc.Log.begin()
parser = ArgumentParser(description="""Gravity wave mountain test by Schar et al (2002).""",
                        add_help=False)

parser.add_argument("--delta",
                    default=1000.0,
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
                    default=18,
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
    tmax = 9000.

delta = args.delta
dt = delta_dt[delta]
hybrid = bool(args.hybridization)
PETSc.Sys.Print("""
Problem parameters:\n
Test case: Gravity wave over a Schar-type mountain.\n
Hybridized compressible solver: %s,\n
delta: %s,\n
Profiling: %s,\n
Test run: %s,\n
Dump frequency: %s.\n
""" % (hybrid, delta, bool(args.profile), bool(args.test), args.dumpfreq))

PETSc.Sys.Print("Initializing problem with dt: %s and tmax: %s.\n" % (dt,
                                                                      tmax))
H = 35000.  # Height position of the model top
L = 144000.
nlayers = int(2*H/delta)  # horizontal layers
columns = int(1.25*L/delta)  # number of columns

PETSc.Sys.Print("Creating mesh with %s columns and %s layers...\n" % (columns,
                                                                      nlayers))
m = PeriodicIntervalMesh(columns, L)
ext_mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

Vc = VectorFunctionSpace(ext_mesh, "DG", 2)
coord = SpatialCoordinate(ext_mesh)
x = Function(Vc).interpolate(as_vector([coord[0], coord[1]]))
a = 1000.
xc = L/2.
x, z = SpatialCoordinate(ext_mesh)
hm = 1.
zs = hm*a**2/((x-xc)**2 + a**2)

smooth_z = True
if smooth_z:
    zh = 5000.
    xexpr = as_vector([x, conditional(z < zh, z + cos(0.5*pi*z/zh)**6*zs, z)])
else:
    xexpr = as_vector([x, z + ((H-z)/H)*zs])
new_coords = Function(Vc).interpolate(xexpr)
mesh = Mesh(new_coords)

# sponge function
W_DG = FunctionSpace(mesh, "DG", 2)
x, z = SpatialCoordinate(mesh)
zc = H-10000.
mubar = 0.15/dt
mu_top = conditional(z <= zc, 0.0, mubar*sin((pi/2.)*(z-zc)/(H-zc))**2)
mu = Function(W_DG).interpolate(mu_top)
fieldlist = ['u', 'rho', 'theta']
timestepping = TimesteppingParameters(dt=dt)

if hybrid:
    dirname = "hybrid_nh_mountain_smootherz_dx%s_dt%s" % (delta, dt)
else:
    dirname = "nh_mountain_smootherz_dx%s_dt%s" % (delta, dt)

output = OutputParameters(dirname=dirname,
                          dumpfreq=args.dumpfreq,
                          dumplist=['u'],
                          perturbation_fields=['theta', 'rho'])
parameters = CompressibleParameters(g=9.80665, cp=1004.)
diagnostics = Diagnostics(*fieldlist)
diagnostic_fields = [CourantNumber(), VelocityZ()]

state = State(mesh, vertical_degree=1, horizontal_degree=1,
              family="CG",
              sponge_function=mu,
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

# Spaces
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

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = 300.
thetab = Tsurf*exp(N**2*z/g)
theta_b = Function(Vt).interpolate(thetab)

# Calculate hydrostatic Pi
PETSc.Sys.Print("Computing hydrostatic Exner pressure...\n")
params = {'pc_type': 'fieldsplit',
          'pc_fieldsplit_type': 'schur',
          'ksp_type': 'gmres',
          'ksp_monitor_true_residual': True,
          'ksp_max_it': 1000,
          'ksp_gmres_restart': 50,
          'pc_fieldsplit_schur_fact_type': 'FULL',
          'pc_fieldsplit_schur_precondition': 'selfp',
          'fieldsplit_0_ksp_type': 'richardson',
          'fieldsplit_0_ksp_max_it': 5,
          'fieldsplit_0_pc_type': 'bjacobi',
          'fieldsplit_0_sub_pc_type': 'ilu',
          'fieldsplit_1_ksp_type': 'richardson',
          'fieldsplit_1_ksp_max_it': 5,
          "fieldsplit_1_ksp_monitor_true_residual": True,
          'fieldsplit_1_pc_type': 'bjacobi',
          'fieldsplit_1_sub_pc_type': 'ilu'}
Pi = Function(Vr)
rho_b = Function(Vr)
compressible_hydrostatic_balance(state,
                                 theta_b,
                                 rho_b,
                                 Pi, top=True,
                                 pi_boundary=0.5,
                                 params=params)


def minimum(f):
    fmin = op2.Global(1, [1000], dtype=float)
    op2.par_loop(op2.Kernel("""
void minify(double *a, double *b) {
    a[0] = a[0] > fabs(b[0]) ? fabs(b[0]) : a[0];
}
""", "minify"), f.dof_dset.set, fmin(op2.MIN), f.dat(op2.READ))
    return fmin.data[0]


p0 = minimum(Pi)
compressible_hydrostatic_balance(state,
                                 theta_b,
                                 rho_b,
                                 Pi,
                                 top=True,
                                 params=params)
p1 = minimum(Pi)
alpha = 2.*(p1-p0)
beta = p1-alpha
pi_top = (1.-beta)/alpha
compressible_hydrostatic_balance(state,
                                 theta_b,
                                 rho_b,
                                 Pi,
                                 top=True,
                                 pi_boundary=pi_top,
                                 solve_for_rho=True,
                                 params=params)

theta0.assign(theta_b)
rho0.assign(rho_b)
u0.project(as_vector([10.0, 0.0]))
remove_initial_w(u0, state.Vv)

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
