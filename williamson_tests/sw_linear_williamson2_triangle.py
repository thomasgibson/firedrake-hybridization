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
    dt = 3600.
    tmax = dt
else:
    # Number of horizontal cells = 20*(4^refinements)
    # Some refinement/dt combinations:
    # refinements = 3, dt = 3600;
    # refinements = 4, dt = 1800;
    # refinements = 5, dt = 900;
    # refinements = 6, dt = 450;
    # refinements = 7, dt = 225.
    # etc.
    refinements = 5
    dt = 900.
    tmax = 5*day

R = 6371220.
H = 2000.

mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=refinements, degree=3)
x = SpatialCoordinate(mesh)
mesh.init_cell_orientations(x)

fieldlist = ['u', 'D']
timestepping = TimesteppingParameters(dt=dt)
output = OutputParameters(dirname='sw_linear_w2',
                          steady_state_error_fields=['u', 'D'])
parameters = ShallowWaterParameters(H=H)
diagnostics = Diagnostics(*fieldlist)

state = State(mesh, horizontal_degree=1,
              family="BDM",
              timestepping=timestepping,
              output=output,
              parameters=parameters,
              diagnostics=diagnostics,
              fieldlist=fieldlist)

# Coriolis expression
Omega = parameters.Omega
x = SpatialCoordinate(mesh)
fexpr = 2*Omega*x[2]/R
V = FunctionSpace(mesh, "CG", 1)
f = state.fields("coriolis", V)
f.interpolate(fexpr)  # Coriolis frequency (1/s)

# interpolate initial conditions
# Initial/current conditions
u0 = state.fields("u")
D0 = state.fields("D")
u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
g = parameters.g
Dexpr = - ((R * Omega * u_max)*(x[2]*x[2]/(R*R)))/g
u0.project(uexpr)
D0.interpolate(Dexpr)
state.initialise([('u', u0),
                  ('D', D0)])

Deqn = LinearAdvection(state, D0.function_space(),
                       state.parameters.H, ibp="once",
                       equation_form="continuity")

advected_fields = []
advected_fields.append(("u", NoAdvection(state, u0, None)))
advected_fields.append(("D", ForwardEuler(state, D0, Deqn)))

# Set up linear solver
linear_solver = HybridizedShallowWaterSolver(state)

# Set up forcing
sw_forcing = ShallowWaterForcing(state, linear=True)

# build time stepper
stepper = CrankNicolson(state, advected_fields, linear_solver,
                        sw_forcing)

stepper.run(t=0, tmax=tmax)
