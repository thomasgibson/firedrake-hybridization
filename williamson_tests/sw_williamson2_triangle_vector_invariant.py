"""
Credit for demo setup: Gusto development team
"""
from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, as_vector, \
    FunctionSpace, parameters
from firedrake.petsc import PETSc
from argparse import ArgumentParser
from hybridization import HybridizedShallowWaterSolver
from math import pi
import sys

day = 24.*60.*60.

ref_dt = {3: 3000.,
          4: 1500.,
          5: 750.,
          6: 375.,
          7: 187.5}

PETSc.Log.begin()
parser = ArgumentParser(description="""Williamson test case 2 (vector invariant).""",
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
                    default=1,
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
    tmax = 5*day

refinements = args.refinements
dt = ref_dt[refinements]
hybrid = bool(args.hybridization)
PETSc.Sys.Print("""
Problem parameters:\n
Test case: Williamson test case 2 (vector invariant)\n
Hybridized shallow water solver: %s,\n
Refinements: %s,\n
Profiling: %s,\n
Test run: %s,\n
Dump frequency: %s.\n
""" % (hybrid, refinements, bool(args.profile),
       bool(args.test), args.dumpfreq))

PETSc.Sys.Print("Initializing problem with dt: %s and tmax: %s.\n" % (dt,
                                                                      tmax))

# setup shallow water parameters
R = 6371220.
H = 5960.

# setup input that doesn't change with ref level or dt
fieldlist = ['u', 'D']
parameters = ShallowWaterParameters(H=H)
diagnostics = Diagnostics(*fieldlist)

dirname = "sw_W2_ref%s_dt%s" % (refinements, dt)
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=refinements, degree=3)
x = SpatialCoordinate(mesh)
global_normal = x
mesh.init_cell_orientations(x)

timestepping = TimesteppingParameters(dt=dt)

if hybrid:
    dirname = 'hybrid_sw_w2_vi_ref%s_dt%s' % (refinements, dt)
else:
    dirname = 'sw_w2_vi_ref%s_dt%s' % (refinements, dt)

output = OutputParameters(dirname=dirname,
                          dumpfreq=args.dumpfreq,
                          dumplist_latlon=['D', 'D_error'],
                          steady_state_error_fields=['D', 'u'])

state = State(mesh, horizontal_degree=1,
              family="BDM",
              timestepping=timestepping,
              output=output,
              parameters=parameters,
              diagnostics=diagnostics,
              fieldlist=fieldlist)

# interpolate initial conditions
u0 = state.fields("u")
D0 = state.fields("D")
x = SpatialCoordinate(mesh)
u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
Omega = parameters.Omega
g = parameters.g
Dexpr = H - ((R * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R*R)))/g
# Coriolis expression
fexpr = 2*Omega*x[2]/R
V = FunctionSpace(mesh, "CG", 1)
f = state.fields("coriolis", V)
f.interpolate(fexpr)  # Coriolis frequency (1/s)

u0.project(uexpr)
D0.interpolate(Dexpr)
state.initialise([('u', u0),
                  ('D', D0)])

ueqn = AdvectionEquation(state, u0.function_space())
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
