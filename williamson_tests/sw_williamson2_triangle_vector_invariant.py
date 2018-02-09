"""
Credit for demo setup: Gusto development team
"""
from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, as_vector, \
    FunctionSpace
from hybridization import HybridizedShallowWaterSolver
from math import pi
import sys

day = 24.*60.*60.

if '--run-test' in sys.argv:
    refinements = 3
    dt = 3000.
    tmax = dt
else:
    # Number of horizontal cells = 20*(4^refinements)
    # Some refinement/dt combinations:
    # refinements = 3, dt = 3000;
    # refinements = 4, dt = 1500;
    # refinements = 5, dt = 750;
    # refinements = 6, dt = 375;
    # refinements = 7, dt = 187.5.
    # etc.
    refinements = 5
    dt = 750.
    tmax = 5*day

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
output = OutputParameters(dirname=dirname,
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
linear_solver = HybridizedShallowWaterSolver(state)

# Set up forcing
sw_forcing = ShallowWaterForcing(state)

# build time stepper
stepper = CrankNicolson(state, advected_fields, linear_solver,
                        sw_forcing)

stepper.run(t=0, tmax=tmax)
