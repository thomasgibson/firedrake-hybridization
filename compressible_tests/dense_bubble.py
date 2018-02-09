"""
Credit for demo setup: Gusto development team
"""
from gusto import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, \
    SpatialCoordinate, Constant, DirichletBC, pi, cos, Function, sqrt, \
    conditional
from hybridization import HybridizedCompressibleSolver
import sys

if '--run-test' in sys.argv:
    dt = 4.
    delta = 800.
    tmax = dt
else:
    # delta/dt combinations:
    # delta = 800., dt = 4;
    # delta = 400., dt = 2;
    # delta = 200., dt = 1;
    # delta = 100., dt = 0.5;
    # delta = 50., dt = 0.25;
    dt = 1.
    delta = 200.
    tmax = 15.*60.

if '--hybrid' in sys.argv:
    hybrid = True
else:
    hybrid = False

L = 51200.

# build volume mesh
H = 6400.  # Height position of the model top

dirname = "db_dx%s_dt%s" % (delta, dt)
nlayers = int(H/delta)  # horizontal layers
columns = int(L/delta)  # number of columns

m = PeriodicIntervalMesh(columns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

fieldlist = ['u', 'rho', 'theta']
timestepping = TimesteppingParameters(dt=dt, maxk=4, maxi=1)
output = OutputParameters(dirname=dirname,
                          dumpfreq=5,
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
compressible_hydrostatic_balance(state, theta_b, rho_b, solve_for_rho=True)

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
    linear_solver = HybridizedCompressibleSolver(state)
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

# build time stepper
stepper = CrankNicolson(state, advected_fields, linear_solver,
                        compressible_forcing, diffused_fields)

stepper.run(t=0, tmax=tmax)
