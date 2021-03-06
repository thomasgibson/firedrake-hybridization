from gusto import *
from firedrake import (CubedSphereMesh, ExtrudedMesh, Expression,
                       VectorFunctionSpace, FunctionSpace, Function,
                       SpatialCoordinate, as_vector, interpolate,
                       CellVolume, exp, acos, cos, sin, pi,
                       sqrt, asin, atan_2, op2)
from firedrake.petsc import PETSc
from argparse import ArgumentParser
import numpy as np
import sys


def fmax(f):
    fmax = op2.Global(1, np.finfo(float).min, dtype=float)
    op2.par_loop(op2.Kernel("""
void maxify(double *a, double *b) {
    a[0] = a[0] < fabs(b[0]) ? fabs(b[0]) : a[0];
}
""", "maxify"), f.dof_dset.set, fmax(op2.MAX), f.dat(op2.READ))
    return fmax.data[0]


PETSc.Log.begin()

parser = ArgumentParser(description=("""
DCMIP Test 3-1 Non-orographic gravity waves on a small planet.
"""), add_help=False)

parser.add_argument("--test",
                    action="store_true",
                    help="Enable a quick test run.")

parser.add_argument("--dt",
                    default=0.0,
                    type=float,
                    action="store",
                    help="Manually set dt")

parser.add_argument("--dumpfreq",
                    # dump every 100s by default
                    default=100,
                    type=int,
                    action="store",
                    help="Dump frequency of output files.")

parser.add_argument("--cfl",
                    default=1.,
                    type=float,
                    action="store",
                    help="CFL number to run at (determines dt).")

parser.add_argument("--refinements",
                    default=4,
                    type=int,
                    action="store",
                    help="Resolution scaling parameter.")

parser.add_argument("--richardson_scale",
                    default=1.0,
                    type=float,
                    action="store",
                    help="Set the Richardson scaling parameter for the trace system.")

parser.add_argument("--flexsolver",
                    action="store_true",
                    help="Switch to flex-GMRES and AMG.")

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

nlayers = args.layers           # Number of vertical layers
refinements = args.refinements  # Number of horiz. cells = 20*(4^refinements)

hybrid = bool(args.hybridization)

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
gamma = (1 - kappa) / kappa
cs = sqrt(c_p * T_eq / gamma)   # Speed of sound in an air parcel

# Cubed-sphere mesh
m = CubedSphereMesh(radius=a,
                    refinement_level=refinements,
                    degree=2)

# Horizontal Courant (advective) number
cell_vs = interpolate(CellVolume(m),
                      FunctionSpace(m, "DG", 0))
a_max = fmax(cell_vs)
dx_max = sqrt(a_max)
u_max = u_0

if args.dt == 0.0:
    cfl = args.cfl
    PETSc.Sys.Print("Determining Dt from specified horizontal CFL: %s" % cfl)
    # Take integer value
    dt = int(cfl * (dx_max / cs))
else:
    dt = args.dt
    cfl = dt * (cs / dx_max)

if args.test:
    tmax = dt
else:
    tmax = 3600.

PETSc.Sys.Print("""
Problem parameters:\n
Test case DCMIP 3-1: Non-orographic gravity waves on a small planet.\n
Hybridized compressible solver: %s,\n
Horizontal refinements: %s,\n
Vertical layers: %s,\n
Max time: %s,\n
Dump frequency: %s,\n
nu CFL: %s.
""" % (hybrid, refinements,
       nlayers,
       tmax,
       int(args.dumpfreq / dt),
       cfl))

PETSc.Sys.Print("Initializing problem with dt: %s and tmax: %s.\n" % (dt,
                                                                      tmax))

# Build volume mesh
z_top = 1.0e4            # Height position of the model top
mesh = ExtrudedMesh(m, layers=nlayers,
                    layer_height=z_top/nlayers,
                    extrusion_type="radial")

# Space for initialising velocity (using this ensures things are in layers)
W_VectorCG1 = VectorFunctionSpace(mesh, "CG", 1)

x = SpatialCoordinate(mesh)

# Create polar coordinates:
# Since we use a CG1 field, this is constant on layers
W_Q1 = FunctionSpace(mesh, "CG", 1)
z_expr = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]) - a
z = Function(W_Q1).interpolate(z_expr)
lat_expr = asin(x[2]/sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]))
lat = Function(W_Q1).interpolate(lat_expr)
lon = Function(W_Q1).interpolate(atan_2(x[1], x[0]))

fieldlist = ['u', 'rho', 'theta']
timestepping = TimesteppingParameters(dt=dt, maxk=4, maxi=1)

dirname = 'meanflow_ref'
if hybrid:
    dirname += '_hybridization'

output = OutputParameters(dumpfreq=int(args.dumpfreq / dt),
                          dirname=dirname,
                          perturbation_fields=['theta', 'rho'],
                          log_level='INFO')

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
# u max defined above
uexpr = as_vector([-u_max*x[1]/a, u_max*x[0]/a, 0.0])
u0.project(uexpr)

# Surface temperature
G = g**2/(N**2*c_p)
Ts_expr = G + (T_eq-G)*exp(-(u_max*N**2/(4*g*g))*u_max*(cos(2.0*lat)-1.0))
Ts = Function(W_Q1).interpolate(Ts_expr)

# Surface pressure
ps_expr = p_eq*exp((u_max/(4.0*G*R_d))*u_max*(cos(2.0*lat)-1.0))*(Ts/T_eq)**(1.0/kappa)
ps = Function(W_Q1).interpolate(ps_expr)

# Background pressure
p_expr = ps*(1 + G/Ts*(exp(-N**2*z/g)-1))**(1.0/kappa)
p = Function(W_Q1).interpolate(p_expr)

# Background temperature
Tb_expr = G*(1 - exp(N**2*z/g)) + Ts*exp(N**2*z/g)
Tb = Function(W_Q1).interpolate(Tb_expr)

# Background potential temperature
thetab_expr = Tb*(p_0/p)**kappa
thetab = Function(W_Q1).interpolate(thetab_expr)
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
thetaeqn = SUPGAdvection(state, Vt,
                         equation_form="advective")
advected_fields = []
advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
advected_fields.append(("rho", SSPRK3(state, rho0, rhoeqn, subcycles=2)))
advected_fields.append(("theta", SSPRK3(state, theta0, thetaeqn, subcycles=2)))

# Set up linear solver
PETSc.Sys.Print("""
Setting up hybridized solver on the traces.""")

if args.flexsolver:
    inner_parameters = {
        'ksp_type': 'fgmres',
        'ksp_rtol': args.rtol,
        'ksp_atol': 1.0e-8,
        'ksp_max_it': 100,
        'pc_type': 'gamg',
        'pc_gamg_sym_graph': True,
        'mg_levels': {
            'ksp_type': 'gmres',
            'ksp_max_it': 5,
            'pc_type': 'bjacobi',
            'sub_pc_type': 'ilu'
        }
    }
else:
    inner_parameters = {
        'ksp_type': 'gmres',
        'ksp_rtol': args.rtol,
        'ksp_max_it': 100,
        'pc_type': 'gamg',
        'mg_levels': {
            'ksp_type': 'richardson',
            'ksp_richardson_scale': args.richardson_scale,
            'pc_type': 'bjacobi',
            'sub_pc_type': 'ilu'
        }
    }
if args.debug:
    inner_parameters['ksp_monitor_true_residual'] = None

# Use Firedrake static condensation interface
solver_parameters = {
    'mat_type': 'matfree',
    'ksp_type': 'preonly',
    'pc_type': 'python',
    'pc_python_type': 'firedrake.SCPC',
    'pc_sc_eliminate_fields': '0, 1',
    'condensed_field': inner_parameters
}

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
stepper = CrankNicolson(state,
                        advected_fields,
                        linear_solver,
                        compressible_forcing)

PETSc.Sys.Print("Starting simulation...\n")
stepper.run(t=0, tmax=tmax)
