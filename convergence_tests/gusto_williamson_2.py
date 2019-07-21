from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       as_vector, FunctionSpace, errornorm)
from math import pi

day = 24.*60.*60.

# setup resolution and timestepping parameters for convergence test
ref_dt = {3: 4000., 4: 2000., 5: 1000., 6: 500.}
tmax = 15*day
dumptime = 5*60.*60.

# setup shallow water parameters
R = 6371220.
H = 5960.

# setup input that doesn't change with ref level or dt
fieldlist = ['u', 'D']
parameters = ShallowWaterParameters(H=H)

for ref_level, dt in ref_dt.items():

    dirname = "sw_W2_ref%s_dt%s" % (ref_level, dt)
    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=ref_level, degree=3)
    x = SpatialCoordinate(mesh)
    global_normal = x
    mesh.init_cell_orientations(x)

    timestepping = TimesteppingParameters(dt=dt)

    dumpfreq = int(dumptime / dt)

    output = OutputParameters(dirname=dirname,
                              dumpfreq=dumpfreq,
                              dumplist_latlon=['D', 'D_error'],
                              steady_state_error_fields=['D', 'u'],
                              log_level='INFO')

    diagnostic_fields = [RelativeVorticity(), PotentialVorticity(),
                         ShallowWaterKineticEnergy(),
                         ShallowWaterPotentialEnergy(),
                         ShallowWaterPotentialEnstrophy()]

    state = State(mesh, horizontal_degree=1,
                  family="BDM",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  diagnostic_fields=diagnostic_fields,
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

    ueqn = VectorInvariant(state, u0.function_space())
    Deqn = AdvectionEquation(state, D0.function_space(),
                             equation_form="continuity")
    advected_fields = []
    advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
    advected_fields.append(("D", SSPRK3(state, D0, Deqn, subcycles=2)))

    linear_solver = ShallowWaterSolver(state)

    # Set up forcing
    sw_forcing = ShallowWaterForcing(state, euler_poincare=False)

    # build time stepper
    stepper = CrankNicolson(state, advected_fields, linear_solver,
                            sw_forcing)

    stepper.run(t=0, tmax=tmax)

    xn = state.xn
    un, Dn = xn.split()

    uerr = errornorm(u0, un, norm_type="L2")
    Derr = errornorm(D0, Dn, norm_type="L2")
