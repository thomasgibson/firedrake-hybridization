"""
Credit for demo setup: Gusto development team
"""
from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, \
    as_vector, pi, sqrt, Min, FunctionSpace
from hybridization import HybridizedShallowWaterSolver
import sys

day = 24.*60.*60.

if '--run-test' in sys.argv:
    refinements = 3
    dt = 3000.
    tmax = dt
else:
    # Number of horizontal cells = 20*(4^refinements)
    # Some refinement/dt combinations:
    # refinements = 3, dt = 900;
    # refinements = 4, dt = 450;
    # refinements = 5, dt = 225;
    # refinements = 6, dt = 112.5;
    # refinements = 7, dt = 56.25.
    # etc.
    refinements = 5
    dt = 225.
    tmax = 50*day

# setup shallow water parameters
R = 6371220.
H = 5960.

# setup input that doesn't change with ref level or dt
fieldlist = ['u', 'D']
parameters = ShallowWaterParameters(H=H)
diagnostics = Diagnostics(*fieldlist)

dirname = "sw_W5_ref%s_dt%s" % (refinements, dt)
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=refinements, degree=3)
x = SpatialCoordinate(mesh)
mesh.init_cell_orientations(x)

timestepping = TimesteppingParameters(dt=dt)
output = OutputParameters(dirname=dirname, dumplist_latlon=['D'], dumpfreq=100)
diagnostic_fields = [Sum('D', 'topography')]

state = State(mesh, horizontal_degree=1,
              family="BDM",
              timestepping=timestepping,
              output=output,
              parameters=parameters,
              diagnostic_fields=diagnostic_fields,
              fieldlist=fieldlist)

# interpolate initial conditions
u0 = state.fields('u')
D0 = state.fields('D')
x = SpatialCoordinate(mesh)
u_max = 20.   # Maximum amplitude of the zonal wind (m/s)
uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
theta, lamda = latlon_coords(mesh)
Omega = parameters.Omega
g = parameters.g
Rsq = R**2
R0 = pi/9.
R0sq = R0**2
lamda_c = -pi/2.
lsq = (lamda - lamda_c)**2
theta_c = pi/6.
thsq = (theta - theta_c)**2
rsq = Min(R0sq, lsq+thsq)
r = sqrt(rsq)
bexpr = 2000 * (1 - r/R0)
Dexpr = H - ((R * Omega * u_max + 0.5*u_max**2)*x[2]**2/Rsq)/g - bexpr

# Coriolis
fexpr = 2*Omega*x[2]/R
V = FunctionSpace(mesh, "CG", 1)
f = state.fields("coriolis", V)
f.interpolate(fexpr)  # Coriolis frequency (1/s)
b = state.fields("topography", D0.function_space())
b.interpolate(bexpr)

u0.project(uexpr)
D0.interpolate(Dexpr)
state.initialise([('u', u0),
                  ('D', D0)])

ueqn = AdvectionEquation(state, u0.function_space(),
                         vector_manifold=True)
Deqn = AdvectionEquation(state, D0.function_space(),
                         equation_form="continuity")

advected_fields = []
advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
advected_fields.append(("D", SSPRK3(state, D0, Deqn)))

# Set up linear solver
linear_solver = HybridizedShallowWaterSolver(state)

# Set up forcing
sw_forcing = ShallowWaterForcing(state, euler_poincare=False)

# build time stepper
stepper = CrankNicolson(state, advected_fields, linear_solver,
                        sw_forcing)

stepper.run(t=0, tmax=tmax)
