"""
Credit for demo setup: Gusto development team
"""
from gusto import *
from firedrake import (as_vector, SpatialCoordinate,
                       PeriodicRectangleMesh, ExtrudedMesh,
                       exp, cos, sin, cosh, sinh, tanh, pi,
                       Function, sqrt)
from firedrake.petsc import PETSc
from argparse import ArgumentParser
import sys


day = 24.*60.*60.
hour = 60.*60.


PETSc.Log.begin()

parser = ArgumentParser(description="""Euler-Boussinesq Eady slice model.""",
                        add_help=False)

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
                    default=50.,
                    type=float,
                    action="store",
                    help="Time step size (seconds)")

parser.add_argument("--res",
                    default=30,
                    type=int,
                    action="store",
                    help="Resolution scaling parameter.")

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

dt = args.dt
res = args.res

if args.profile:
    tmax = 20*dt

if args.test:
    tmax = dt

if not args.test and not args.profile:
    tmax = 30*day


hybrid = bool(args.hybridization)
PETSc.Sys.Print("""
Problem parameters:\n
Test case: Euler-Boussinesq Eady model.\n
Hybridized compressible solver: %s,\n
Profiling: %s,\n
Test run: %s.\n
""" % (hybrid, bool(args.profile), bool(args.test)))

PETSc.Sys.Print("Initializing problem with dt: %s and tmax: %s.\n" % (dt,
                                                                      tmax))


# Construct 1d periodic base mesh
columns = 2*res  # number of columns
L = 1000000.
m = PeriodicRectangleMesh(columns, 1, 2.*L, 1.e5, quadrilateral=True)

# Build 2D mesh by extruding the base mesh
nlayers = res    # horizontal layers
H = 10000.  # height position of the model top
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

# Set up all the other things that state requires:
# Coriolis expression
f = 1.e-04
Omega = as_vector([0., 0., f*0.5])

# List of prognostic fieldnames:
# this is passed to state and used to construct a dictionary,
# state.field_dict so that we can access fields by name.
# u is the 3D velocity
# p is the pressure
# b is the buoyancy
fieldlist = ['u', 'rho', 'theta']

# Class containing timestepping parameters:
# all values not explicitly set here use the default values provided
# and documented in configuration.py
timestepping = TimesteppingParameters(dt=dt)

# Class containing output parameters:
# all values not explicitly set here use the default values provided
# and documented in configuration.py
if hybrid:
    dirname = "hybrid_compressible_eady"
else:
    dirname = "compressible_eady"

output = OutputParameters(dirname=dirname,
                          dumpfreq=int(2*hour/dt),
                          dumplist=['u', 'rho', 'theta'],
                          perturbation_fields=['rho', 'theta', 'ExnerPi'])

# Class containing physical parameters:
# all values not explicitly set here use the default values provided
# and documented in configuration.py
parameters = CompressibleEadyParameters(H=H, f=f)

# Class for diagnostics:
# fields passed to this class will have basic diagnostics computed
# (eg min, max, l2 norm) and these will be output as a json file
diagnostics = Diagnostics(*fieldlist)

# List of diagnostic fields, each defined in a class in diagnostics.py
diagnostic_fields = [CourantNumber(),
                     VelocityY(),
                     ExnerPi(),
                     ExnerPi(reference=True),
                     CompressibleKineticEnergy(),
                     CompressibleKineticEnergyY(),
                     CompressibleEadyPotentialEnergy(),
                     Sum("CompressibleKineticEnergy",
                         "CompressibleEadyPotentialEnergy"),
                     Difference("CompressibleKineticEnergy",
                                "CompressibleKineticEnergyY")]

# Setup state, passing in the mesh, information on the required finite element
# function spaces and the classes above
state = State(mesh,
              vertical_degree=1,
              horizontal_degree=1,
              family="RTCF",
              Coriolis=Omega,
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

# First setup the background buoyancy profile
# z.grad(bref) = N**2.
# The following is symbolic algebra, using the default buoyancy frequency
# from the parameters class.
x, y, z = SpatialCoordinate(mesh)
g = parameters.g
Nsq = parameters.Nsq
theta_surf = parameters.theta_surf

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
theta_ref = theta_surf*exp(Nsq*(z-H/2)/g)
theta_b = Function(Vt).interpolate(theta_ref)


# Set theta_pert
def coth(x):
    return cosh(x)/sinh(x)


def Z(z):
    return Bu*((z/H)-0.5)


def n():
    return Bu**(-1)*sqrt((Bu*0.5-tanh(Bu*0.5))*(coth(Bu*0.5)-Bu*0.5))


a = -4.5
Bu = 0.5
theta_exp = a*theta_surf/g*sqrt(Nsq)*(-(1.-Bu*0.5*coth(Bu*0.5))*sinh(Z(z))*cos(pi*(x-L)/L)
                                      - n()*Bu*cosh(Z(z))*sin(pi*(x-L)/L))
theta_pert = Function(Vt).interpolate(theta_exp)

# Set theta0
theta0.interpolate(theta_b + theta_pert)

# Calculate hydrostatic Pi
PETSc.Sys.Print("Computing hydrostatic varaibles...\n")

# Use vertical hybridization preconditioner for the balance initialization
pi_params = {
    'ksp_type': 'preonly',
    'pc_type': 'python',
    'mat_type': 'matfree',
    'pc_python_type': 'gusto.VerticalHybridizationPC',
    'vert_hybridization': {
        'ksp_type': 'gmres',
        'pc_type': 'gamg',
        'pc_gamg_sym_graph': True,
        'ksp_rtol': 1e-12,
        'ksp_atol': 1e-12,
        'mg_levels': {
            'ksp_type': 'richardson',
            'ksp_max_it': 3,
            'pc_type': 'bjacobi',
            'sub_pc_type': 'ilu'
        }
    }
}
if args.debug:
    pi_params['vert_hybridization']['ksp_monitor_true_residual'] = True

rho_b = Function(Vr)
compressible_hydrostatic_balance(state,
                                 theta_b,
                                 rho_b,
                                 params=pi_params)
compressible_hydrostatic_balance(state,
                                 theta0,
                                 rho0,
                                 params=pi_params)

# Set Pi0
Pi0 = calculate_Pi0(state, theta0, rho0)
state.parameters.Pi0 = Pi0

# Set x component of velocity
cp = state.parameters.cp
dthetady = state.parameters.dthetady
Pi = thermodynamics.pi(state.parameters, rho0, theta0)
u = cp*dthetady/f*(Pi-Pi0)

# Set y component of velocity
v = Function(Vr).assign(0.)
compressible_eady_initial_v(state, theta0, rho0, v)

# Set initial u
u_exp = as_vector([u, v, 0.])
u0.project(u_exp)

# Pass these initial conditions to the state.initialise method
state.initialise([('u', u0),
                  ('rho', rho0),
                  ('theta', theta0)])

# Set the background profiles
state.set_reference_profiles([('rho', rho_b),
                              ('theta', theta_b)])

# Set up advection schemes
# we need a DG funciton space for the embedded DG advection scheme
ueqn = AdvectionEquation(state, Vu)
rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
thetaeqn = SUPGAdvection(state, Vt, supg_params={"dg_direction": "horizontal"})

advected_fields = []
advected_fields.append(("u", SSPRK3(state, u0, ueqn)))
advected_fields.append(("rho", SSPRK3(state, rho0, rhoeqn)))
advected_fields.append(("theta", SSPRK3(state, theta0, thetaeqn)))

# Set up linear solver
if hybrid:
    inner_parameters = {
        'ksp_type': 'fgmres',
        'ksp_rtol': 1.0e-8,
        'ksp_atol': 1.0e-8,
        'ksp_max_it': 100,
        'pc_type': 'gamg',
        'pc_gamg_sym_graph': True,
        'mg_levels': {
            'ksp_type': 'gmres',
            'ksp_max_its': 5,
            'pc_type': 'bjacobi',
            'sub_pc_type': 'ilu'
        }
    }
    if args.debug:
        inner_parameters['ksp_monitor_true_residual'] = True

    # Use Firedrake static condensation interface
    solver_parameters = {
        'mat_type': 'matfree',
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.SCPC',
        'pc_sc_eliminate_fields': '0, 1',
        'condensed_field': inner_parameters
    }
    linear_solver = HybridizedCompressibleSolver(state,
                                                 solver_parameters=solver_parameters,
                                                 overwrite_solver_parameters=True)

else:
    linear_solver_params = {
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'schur',
        'ksp_type': 'gmres',
        'ksp_max_it': 100,
        'ksp_gmres_restart': 50,
        'pc_fieldsplit_schur_fact_type': 'FULL',
        'pc_fieldsplit_schur_precondition': 'selfp',
        'fieldsplit_0': {
            'ksp_type': 'preonly',
            'pc_type': 'bjacobi',
            'sub_pc_type': 'ilu'
        },
        'fieldsplit_1': {
            'ksp_type': 'preonly',
            'pc_type': 'gamg',
            'pc_gamg_sym_graph': True,
            'mg_levels': {
                'ksp_type': 'chebyshev',
                'ksp_chebyshev_esteig': True,
                'ksp_max_it': 5,
                'pc_type': 'bjacobi',
                'sub_pc_type': 'ilu'
            }
        }
    }
    if args.debug:
        linear_solver_params['ksp_monitor_true_residual'] = True

    linear_solver = CompressibleSolver(state,
                                       solver_parameters=linear_solver_params,
                                       overwrite_solver_parameters=True)

# Set up forcing
forcing = CompressibleEadyForcing(state, euler_poincare=False)

# Build time stepper
stepper = CrankNicolson(state, advected_fields, linear_solver, forcing)

PETSc.Sys.Print("Starting simulation...\n")
stepper.run(t=0, tmax=tmax)
