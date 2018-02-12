"""
Credit for demo setup: Gusto development team
"""
from gusto import *
from firedrake import IcosahedralSphereMesh, cos, sin, SpatialCoordinate, \
    FunctionSpace, parameters
from firedrake.petsc import PETSc
from argparse import ArgumentParser
from hybridization import HybridizedShallowWaterSolver
import sys

day = 24.*60.*60.

ref_dt = {3: 1800.,
          4: 900.,
          5: 450.,
          6: 225.,
          7: 112.5}

PETSc.Log.begin()
parser = ArgumentParser(description="""Williamson test case 6 (triangles).""",
                        add_help=False)

parser.add_argument("--refinements",
                    default=3,
                    type=int,
                    choices=[3, 4, 5, 6, 7],
                    help="Number of icosahedral refinements.")

parser.add_argument("--hybridization",
                    action="store_true",
                    help="Use a hybridized shallow water solver.")

parser.add_argument("--test",
                    action="store_true",
                    help="Enable a quick test run.")

parser.add_argument("--profile",
                    action="store_true",
                    help="Turn on profiling for a 20 time-step run.")

parser.add_argument("--dumpfreq",
                    default=24,
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
    tmax = 20*ref_dt[args.refinements]

if args.test:
    tmax = ref_dt[args.refinements]

if not args.test and not args.profile:
    tmax = 14*day

refinements = args.refinements
dt = ref_dt[refinements]
hybrid = bool(args.hybridization)
PETSc.Sys.Print("""
Problem parameters:\n
Test case: Williamson test case 6 (triangles)\n
Hybridized shallow water solver: %s,\n
Refinements: %s,\n
Profiling: %s,\n
Test run: %s,\n
Dump frequency: %s.\n
""" % (hybrid, refinements, bool(args.profile),
       bool(args.test), args.dumpfreq))

PETSc.Sys.Print("Initializing problem with dt: %s and tmax: %s.\n" % (dt,
                                                                      tmax))

R = 6371220.
H = 8000.

mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=refinements)
x = SpatialCoordinate(mesh)
mesh.init_cell_orientations(x)

fieldlist = ['u', 'D']
timestepping = TimesteppingParameters(dt=dt)

if hybrid:
    dirname = 'hybrid_sw_rossby_wave_ll_ref%s_dt%s' % (refinements, dt)
else:
    dirname = 'sw_rossby_wave_ll_ref%s_dt%s' % (refinements, dt)

output = OutputParameters(dirname=dirname,
                          dumpfreq=args.dumpfreq,
                          dumplist_latlon=['D'])
parameters = ShallowWaterParameters(H=H)
diagnostics = Diagnostics(*fieldlist)
diagnostic_fields = [CourantNumber()]

state = State(mesh, horizontal_degree=1,
              family="BDM",
              timestepping=timestepping,
              output=output,
              parameters=parameters,
              diagnostics=diagnostics,
              fieldlist=fieldlist,
              diagnostic_fields=diagnostic_fields)

# interpolate initial conditions
# Initial/current conditions
u0 = state.fields("u")
D0 = state.fields("D")
omega = 7.848e-6  # note lower-case, not the same as Omega
K = 7.848e-6
g = parameters.g
Omega = parameters.Omega

theta, lamda = latlon_coords(mesh)

u_zonal = R*omega*cos(theta) + R*K*(cos(theta)**3)*(4*sin(theta)**2 - cos(theta)**2)*cos(4*lamda)
u_merid = -R*K*4*(cos(theta)**3)*sin(theta)*sin(4*lamda)

uexpr = sphere_to_cartesian(mesh, u_zonal, u_merid)


def Atheta(theta):
    return 0.5*omega*(2*Omega + omega)*cos(theta)**2 + 0.25*(K**2)*(cos(theta)**8)*(5*cos(theta)**2 + 26 - 32/(cos(theta)**2))


def Btheta(theta):
    return (2*(Omega + omega)*K/30)*(cos(theta)**4)*(26 - 25*cos(theta)**2)


def Ctheta(theta):
    return 0.25*(K**2)*(cos(theta)**8)*(5*cos(theta)**2 - 6)


Dexpr = H + (R**2)*(Atheta(theta) + Btheta(theta)*cos(4*lamda) + Ctheta(theta)*cos(8*lamda))/g

# Coriolis
fexpr = 2*Omega*x[2]/R
V = FunctionSpace(mesh, "CG", 1)
f = state.fields("coriolis", V)
f.interpolate(fexpr)  # Coriolis frequency (1/s)

u0.project(uexpr, form_compiler_parameters={'quadrature_degree': 8})
D0.interpolate(Dexpr)

state.initialise([('u', u0),
                  ('D', D0)])

ueqn = EulerPoincare(state, u0.function_space())
Deqn = AdvectionEquation(state, D0.function_space(),
                         equation_form="continuity")

advected_fields = []
advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
advected_fields.append(("D", SSPRK3(state, D0, Deqn)))

# Set up linear solver
if hybrid:
    linear_solver = HybridizedShallowWaterSolver(state)

else:
    parameters = {'ksp_type': 'gmres',
                  'pc_type': 'fieldsplit',
                  'pc_fieldsplit_type': 'schur',
                  'ksp_type': 'gmres',
                  'ksp_max_it': 100,
                  'ksp_gmres_restart': 50,
                  'pc_fieldsplit_schur_fact_type': 'FULL',
                  'pc_fieldsplit_schur_precondition': 'selfp',
                  'fieldsplit_0': {'ksp_type': 'preonly',
                                   'pc_type': 'bjacobi',
                                   'sub_pc_type': 'ilu'},
                  'fieldsplit_1': {'ksp_type': 'cg',
                                   'pc_type': 'gamg',
                                   'ksp_rtol': 1e-8,
                                   'mg_levels': {'ksp_type': 'chebyshev',
                                                 'ksp_max_it': 2,
                                                 'pc_type': 'bjacobi',
                                                 'sub_pc_type': 'ilu'}}}

    # Shallow water solver from Gusto, but with hybridization turned off
    linear_solver = ShallowWaterSolver(state, solver_parameters=parameters,
                                       overwrite_solver_parameters=True)

# Set up forcing
sw_forcing = ShallowWaterForcing(state)

# Build time stepper
stepper = CrankNicolson(state, advected_fields, linear_solver,
                        sw_forcing)

PETSc.Sys.Print("Starting simulation...\n")
stepper.run(t=0, tmax=tmax)
