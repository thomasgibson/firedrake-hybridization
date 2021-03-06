from gusto import *
from firedrake import (CubedSphereMesh, ExtrudedMesh, IcosahedralSphereMesh,
                       FunctionSpace, Function, VectorFunctionSpace,
                       SpatialCoordinate, interpolate,
                       CellVolume, exp, acos, cos, sin,
                       sqrt, asin, atan_2, op2)
from firedrake.petsc import PETSc
from collections import namedtuple
from profiler import Profiler
import numpy as np

np.random.seed(2097152)


PETSc.Log.begin()


__all__ = ["run_profliler"]


# Container object for storing parameter related information
ParameterInfo = namedtuple("ParameterInfo",
                           ["dt",
                            "deltax",
                            "deltaz",
                            "horizontal_courant",
                            "vertical_courant",
                            "family",
                            "model_degree",
                            "mesh_degree",
                            "solver_type",
                            "inner_solver_type"])


def fmax(f):
    fmax = op2.Global(1, np.finfo(float).min, dtype=float)
    op2.par_loop(op2.Kernel("""
void maxify(double *a, double *b) {
    a[0] = a[0] < fabs(b[0]) ? fabs(b[0]) : a[0];
}
""", "maxify"), f.dof_dset.set, fmax(op2.MAX), f.dat(op2.READ))
    return fmax.data[0]


def run_profliler(hybridization, model_degree, model_family, mesh_degree,
                  cfl, refinements, layers, debug, rtol,
                  flexsolver=True, stronger_smoother=False,
                  suppress_data_output=False):

    nlayers = layers           # Number of vertical layers
    refinements = refinements  # Number of horiz. cells = 20*(4^refinements)

    hybrid = bool(hybridization)

    # Set up problem parameters
    parameters = CompressibleParameters()
    a_ref = 6.37122e6               # Radius of the Earth (m)
    X = 1.0                         # Reduced-size Earth reduction factor
    a = a_ref/X                     # Scaled radius of planet (m)
    g = parameters.g                # Acceleration due to gravity (m/s^2)
    N = parameters.N                # Brunt-Vaisala frequency (1/s)
    p_0 = parameters.p_0            # Reference pressure (Pa, not hPa)
    c_p = parameters.cp             # SHC of dry air at const. pressure (J/kg/K)
    R_d = parameters.R_d            # Gas constant for dry air (J/kg/K)
    kappa = parameters.kappa        # R_d/c_p
    T_eq = 300.0                    # Isothermal atmospheric temperature (K)
    p_eq = 1000.0 * 100.0           # Reference surface pressure at the equator
    u_0 = 20.0                      # Maximum amplitude of the zonal wind (m/s)
    d = 5000.0                      # Width parameter for Theta'
    lamda_c = 2.0*np.pi/3.0         # Longitudinal centerpoint of Theta'
    phi_c = 0.0                     # Lat. centerpoint of Theta' (equator)
    deltaTheta = 1.0                # Maximum amplitude of Theta' (K)
    L_z = 20000.0                   # Vert. wave length of the Theta' perturb.
    gamma = (1 - kappa) / kappa
    cs = sqrt(c_p * T_eq / gamma)   # Speed of sound in an air parcel

    if model_family == "RTCF":
        # Cubed-sphere mesh
        m = CubedSphereMesh(radius=a,
                            refinement_level=refinements,
                            degree=mesh_degree)
    elif model_family == "RT" or model_family == "BDFM":
        m = IcosahedralSphereMesh(radius=a,
                                  refinement_level=refinements,
                                  degree=mesh_degree)
    else:
        raise ValueError("Unknown family: %s" % model_family)

    cell_vs = interpolate(CellVolume(m),
                          FunctionSpace(m, "DG", 0))

    a_max = fmax(cell_vs)
    dx_max = sqrt(a_max)
    u_max = u_0

    PETSc.Sys.Print("\nDetermining Dt from specified horizontal CFL: %s" % cfl)
    dt = int(cfl * (dx_max / cs))

    # Height position of the model top (m)
    z_top = 1.0e4
    deltaz = z_top / nlayers

    vertical_cfl = dt * (cs / deltaz)

    PETSc.Sys.Print("""
Problem parameters:\n
Profiling linear solver for the compressible Euler equations.\n
Speed of sound in compressible atmosphere: %s,\n
Hybridized compressible solver: %s,\n
Model degree: %s,\n
Model discretization: %s,\n
Mesh degree: %s,\n
Horizontal refinements: %s,\n
Vertical layers: %s,\n
Dx (max, m): %s,\n
Dz (m): %s,\n
Dt (s): %s,\n
horizontal CFL: %s,\n
vertical CFL: %s.
""" % (cs,
       hybrid,
       model_degree,
       model_family,
       mesh_degree,
       refinements,
       nlayers,
       dx_max,
       deltaz,
       dt,
       cfl,
       vertical_cfl))

    # Build volume mesh
    mesh = ExtrudedMesh(m, layers=nlayers,
                        layer_height=deltaz,
                        extrusion_type="radial")

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
    timestepping = TimesteppingParameters(dt=dt, maxk=1, maxi=1)

    dirname = 'meanflow_ref'
    if hybrid:
        dirname += '_hybridization'

    # No output
    output = OutputParameters(dumpfreq=3600,
                              dirname=dirname,
                              perturbation_fields=['theta', 'rho'],
                              dump_vtus=False,
                              dump_diagnostics=False,
                              checkpoint=False,
                              log_level='INFO')

    diagnostics = Diagnostics(*fieldlist)

    state = State(mesh,
                  vertical_degree=model_degree,
                  horizontal_degree=model_degree,
                  family=model_family,
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

    x = SpatialCoordinate(mesh)

    # Random velocity field
    CG2 = VectorFunctionSpace(mesh, "CG", 2)
    urand = Function(CG2)
    urand.dat.data[:] += np.random.randn(*urand.dat.data.shape)
    u0.project(urand)

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
    theta0.interpolate(theta_pert)

    # Compute the balanced density
    PETSc.Sys.Print("Computing balanced density field...\n")

    # Use vert. hybridization preconditioner for initialization
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
    if debug:
        pi_params['vert_hybridization']['ksp_monitor_true_residual'] = None

    compressible_hydrostatic_balance(state,
                                     theta_b,
                                     rho_b,
                                     top=False,
                                     pi_boundary=(p/p_0)**kappa,
                                     solve_for_rho=False,
                                     params=pi_params)

    # Random potential temperature perturbation
    theta0.assign(0.0)
    theta0.dat.data[:] += np.random.randn(len(theta0.dat.data))

    # Random density field
    rho0.assign(0.0)
    rho0.dat.data[:] += np.random.randn(len(rho0.dat.data))

    state.initialise([('u', u0),
                      ('rho', rho0),
                      ('theta', theta0)])
    state.set_reference_profiles([('rho', rho_b),
                                  ('theta', theta_b)])

    # Set up advection schemes
    ueqn = EulerPoincare(state, Vu)
    rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
    thetaeqn = SUPGAdvection(state, Vt,
                             equation_form="advective")
    advected_fields = []
    advected_fields.append(("u",
                            ThetaMethod(state, u0, ueqn)))
    advected_fields.append(("rho",
                            SSPRK3(state, rho0, rhoeqn, subcycles=2)))
    advected_fields.append(("theta",
                            SSPRK3(state, theta0, thetaeqn, subcycles=2)))

    # Set up linear solver
    if hybrid:

        outer_solver_type = "Hybrid_SCPC"

        PETSc.Sys.Print("""
Setting up hybridized solver on the traces.""")

        if flexsolver:

            inner_solver_type = "fgmres_gamg_gmres_smoother"

            inner_parameters = {
                'ksp_type': 'fgmres',
                'ksp_rtol': rtol,
                'ksp_max_it': 500,
                'ksp_gmres_restart': 30,
                'pc_type': 'gamg',
                'pc_gamg_sym_graph': None,
                'mg_levels': {
                    'ksp_type': 'gmres',
                    'pc_type': 'bjacobi',
                    'sub_pc_type': 'ilu',
                    'ksp_max_it': 3
                }
            }

        else:

            inner_solver_type = "fgmres_ml_richardson"

            inner_parameters = {
                'ksp_type': 'fgmres',
                'ksp_rtol': rtol,
                'ksp_max_it': 500,
                'ksp_gmres_restart': 30,
                'pc_type': 'ml',
                'pc_mg_cycles': 1,
                'pc_ml_maxNlevels': 25,
                'mg_levels': {
                    'ksp_type': 'richardson',
                    'ksp_richardson_scale': 0.8,
                    'pc_type': 'bjacobi',
                    'sub_pc_type': 'ilu',
                    'ksp_max_it': 3
                }
            }

        if stronger_smoother:
            inner_parameters['mg_levels']['ksp_max_it'] = 5
            inner_solver_type += "_stronger"

        if debug:
            PETSc.Sys.Print("""Debugging on.""")
            inner_parameters['ksp_monitor_true_residual'] = None

        PETSc.Sys.Print("Inner solver: %s" % inner_solver_type)

        # Use Firedrake static condensation interface
        solver_parameters = {
            'mat_type': 'matfree',
            'pmat_type': 'matfree',
            'ksp_type': 'preonly',
            'pc_type': 'python',
            'pc_python_type': 'firedrake.SCPC',
            'pc_sc_eliminate_fields': '0, 1',
            'condensed_field': inner_parameters
        }

        linear_solver = HybridizedCompressibleSolver(
            state,
            solver_parameters=solver_parameters,
            overwrite_solver_parameters=True
        )

    else:

        outer_solver_type = "gmres_SchurPC"

        PETSc.Sys.Print("""
Setting up GCR fieldsplit solver with Schur complement PC.""")

        solver_parameters = {
                'pc_type': 'fieldsplit',
                'pc_fieldsplit_type': 'schur',
                'ksp_type': 'fgmres',
                'ksp_max_it': 100,
                'ksp_rtol': rtol,
                'pc_fieldsplit_schur_fact_type': 'FULL',
                'pc_fieldsplit_schur_precondition': 'selfp',
                'fieldsplit_0': {
                    'ksp_type': 'preonly',
                    'pc_type': 'bjacobi',
                    'sub_pc_type': 'ilu'
                },
                'fieldsplit_1': {
                    'ksp_type': 'preonly',
                    'ksp_max_it': 30,
                    'ksp_monitor_true_residual': None,
                    'pc_type': 'hypre',
                    'pc_hypre_type': 'boomeramg',
                    'pc_hypre_boomeramg_max_iter': 1,
                    'pc_hypre_boomeramg_agg_nl': 0,
                    'pc_hypre_boomeramg_coarsen_type': 'Falgout',
                    'pc_hypre_boomeramg_smooth_type': 'Euclid',
                    'pc_hypre_boomeramg_eu_bj': 1,
                    'pc_hypre_boomeramg_interptype': 'classical',
                    'pc_hypre_boomeramg_P_max': 0,
                    'pc_hypre_boomeramg_agg_nl': 0,
                    'pc_hypre_boomeramg_strong_threshold': 0.25,
                    'pc_hypre_boomeramg_max_levels': 25,
                    'pc_hypre_boomeramg_no_CF': False
                }
            }

        inner_solver_type = "hypre"

        if debug:
            solver_parameters['ksp_monitor_true_residual'] = None

        linear_solver = CompressibleSolver(
            state,
            solver_parameters=solver_parameters,
            overwrite_solver_parameters=True
        )

    # Set up forcing
    compressible_forcing = CompressibleForcing(state)

    param_info = ParameterInfo(dt=dt,
                               deltax=dx_max,
                               deltaz=deltaz,
                               horizontal_courant=cfl,
                               vertical_courant=vertical_cfl,
                               family=model_family,
                               model_degree=model_degree,
                               mesh_degree=mesh_degree,
                               solver_type=outer_solver_type,
                               inner_solver_type=inner_solver_type)

    # Build profiler
    profiler = Profiler(parameterinfo=param_info,
                        state=state,
                        advected_fields=advected_fields,
                        linear_solver=linear_solver,
                        forcing=compressible_forcing,
                        suppress_data_output=suppress_data_output)

    PETSc.Sys.Print("Starting profiler...\n")
    profiler.run(t=0, tmax=dt)
