from firedrake import *
from firedrake.petsc import PETSc
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import sys


ref_to_dt = {3: 3000.0,
             4: 1500.0,
             5: 750.0,
             6: 375.0,
             7: 187.5,
             8: 93.75}


PETSc.Log.begin()
parser = ArgumentParser(description="""Run convergence test for Williamson test case 2""",
                        add_help=False)

parser.add_argument("--hybridization",
                    action="store_true",
                    help="Turn hybridization on.")

parser.add_argument("--verbose",
                    action="store_true",
                    help="Turn on energy and output print statements.")

parser.add_argument("--model_degree",
                    action="store",
                    type=int,
                    default=2,
                    help="Degree of the finite element model.")

parser.add_argument("--test",
                    action="store_true",
                    help=("Select 'True' or 'False' to enable a test run. "
                          "Default is False."))

parser.add_argument("--dumpfreq",
                    default=10,
                    type=int,
                    action="store",
                    help="Dump frequency of output.")

parser.add_argument("--help",
                    action="store_true",
                    help="Show help")

args, _ = parser.parse_known_args()

if args.help:
    help = parser.format_help()
    PETSc.Sys.Print("%s\n" % help)
    sys.exit(1)


def run_williamson2(refinement_level, dumpfreq=100, test=False,
                    verbose=True, model_degree=2, hybridization=False):

    if refinement_level not in ref_to_dt:
        raise ValueError("Refinement level must be one of "
                         "the following: [3, 4, 5, 6, 7, 8]")

    Dt = ref_to_dt[refinement_level]
    R = 6371220.
    H = Constant(5960.)
    day = 24.*60.*60.

    # Earth-sized mesh
    # mesh_degree = 2
    # mesh = OctahedralSphereMesh(R, refinement_level,
    #                             degree=mesh_degree,
    #                             hemisphere="both")
    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=refinement_level,
                                 degree=3)

    global_normal = Expression(("x[0]", "x[1]", "x[2]"))
    mesh.init_cell_orientations(global_normal)

    x = SpatialCoordinate(mesh)

    # Maximum amplitude of zonal winds (m/s)
    u_0 = 2*pi*R/(12*day)
    bexpr = Expression("0.")  # No topography

    if test:
        tmax = 5*Dt
        PETSc.Sys.Print("Taking 5 time-steps\n")
    else:
        tmax = 5*day
        PETSc.Sys.Print("Running 5 day solid-body simulation\n")

    # Compatible FE spaces for velocity and depth
    Vu = FunctionSpace(mesh, "BDM", model_degree)
    VD = FunctionSpace(mesh, "DG", model_degree - 1)

    # State variables: velocity and depth
    un = Function(Vu, name="Velocity")
    Dn = Function(VD, name="Depth")

    outward_normals = CellNormal(mesh)

    def perp(u):
        return cross(outward_normals, u)

    # Initial conditions for velocity and depth (in geostrophic balance)
    u_max = Constant(u_0)
    R0 = Constant(R)
    uexpr = as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])
    h0 = Constant(H)
    Omega = Constant(7.292e-5)
    g = Constant(9.810616)
    Dexpr = h0 - ((R0 * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R0*R0)))/g
    Dn.interpolate(Dexpr)
    un.project(uexpr)
    b = Function(VD, name="Topography").interpolate(bexpr)
    Dn -= b

    # Coriolis expression (1/s)
    fexpr = 2*Omega*x[2]/R0
    Vm = FunctionSpace(mesh, "CG", mesh_degree)
    f = Function(Vm).interpolate(fexpr)

    # Build timestepping solver
    up = Function(Vu)
    Dp = Function(VD)
    dt = Constant(Dt)

    # Stage 1: Depth advection
    # DG upwinded advection for depth
    Dps = Function(VD)
    D = TrialFunction(VD)
    phi = TestFunction(VD)
    Dh = 0.5*(Dn + D)
    uh = 0.5*(un + up)
    n = FacetNormal(mesh)
    uup = 0.5*(dot(uh, n) + abs(dot(uh, n)))

    Deqn = (
        (D - Dn)*phi*dx - dt*inner(grad(phi), uh*Dh)*dx
        + dt*jump(phi)*(uup('+')*Dh('+')-uup('-')*Dh('-'))*dS
    )

    Dproblem = LinearVariationalProblem(lhs(Deqn), rhs(Deqn), Dps)
    Dsolver = LinearVariationalSolver(Dproblem,
                                      solver_parameters={'ksp_type': 'cg',
                                                         'pc_type': 'bjacobi',
                                                         'sub_pc_type': 'ilu'},
                                      options_prefix="D-advection")

    # Stage 2: U update
    Ups = Function(Vu)
    u = TrialFunction(Vu)
    v = TestFunction(Vu)
    Dh = 0.5*(Dn + Dp)
    ubar = 0.5*(un + up)
    uup = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))
    uh = 0.5*(un + u)
    Upwind = 0.5*(sign(dot(ubar, n)) + 1)

    # Kinetic energy term (implicit midpoint)
    K = 0.5*(inner(0.5*(un + up), 0.5*(un + up)))
    both = lambda u: 2*avg(u)
    # u_t + gradperp.u + f)*perp(ubar) + grad(g*D + K)
    # <w, gradperp.u * perp(ubar)> = <perp(ubar).w, gradperp(u)>
    #                                = <-gradperp(w.perp(ubar))), u>
    #                                  +<< [[perp(n)(w.perp(ubar))]], u>>
    ueqn = (
        inner(u - un, v)*dx + dt*inner(perp(uh)*f, v)*dx
        - dt*inner(perp(grad(inner(v, perp(ubar)))), uh)*dx
        + dt*inner(both(perp(n)*inner(v, perp(ubar))), both(Upwind*uh))*dS
        - dt*div(v)*(g*(Dh + b) + K)*dx
    )

    Uproblem = LinearVariationalProblem(lhs(ueqn), rhs(ueqn), Ups)
    Usolver = LinearVariationalSolver(Uproblem,
                                      solver_parameters={'ksp_type': 'gmres',
                                                         'pc_type': 'bjacobi',
                                                         'sub_pc_type': 'ilu'},
                                      options_prefix="U-advection")

    # Stage 3: Implicit linear solve for u, D increments
    W = MixedFunctionSpace((Vu, VD))
    DU = Function(W)
    w, phi = TestFunctions(W)
    du, dD = split(DU)

    uDlhs = (
        inner(w, du + 0.5*dt*f*perp(du)) - 0.5*dt*div(w)*g*dD +
        phi*(dD + 0.5*dt*H*div(du))
    )*dx
    Dh = 0.5*(Dp + Dn)
    uh = 0.5*(un + up)

    uDrhs = -(
        inner(w, up - Ups)*dx
        + phi*(Dp - Dps)*dx
    )

    FuD = uDlhs - uDrhs
    DUproblem = NonlinearVariationalProblem(FuD, DU)

    if hybridization:
        parameters = {'snes_type': 'ksponly',
                      'ksp_type': 'preonly',
                      'pmat_type': 'matfree',
                      'pc_type': 'python',
                      'pc_python_type': 'scpc.HybridizationPC',
                      'hybridization': {'ksp_type': 'cg',
                                        'pc_type': 'gamg',
                                        'ksp_rtol': 1e-8,
                                        'mg_levels': {'ksp_type': 'chebyshev',
                                                      'ksp_max_it': 2,
                                                      'pc_type': 'bjacobi',
                                                      'sub_pc_type': 'ilu'}}}

    else:
        parameters = {'snes_type': 'ksponly',
                      'ksp_type': 'gmres',
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

    DUsolver = NonlinearVariationalSolver(DUproblem,
                                          solver_parameters=parameters,
                                          options_prefix="implicit-solve")
    deltau, deltaD = DU.split()

    dumpcount = dumpfreq
    count = 0
    dirname = "results-W2/"
    if hybridization:
        dirname += "hybrid/"
    dirname += "refinement_" + str(refinement_level) + "/"
    Dfile = File(dirname + "w2_" + str(refinement_level) + ".pvd")
    eta = Function(VD, name="Surface Height")

    def dump(dumpcount, dumpfreq, count):
        dumpcount += 1
        if(dumpcount > dumpfreq):
            if verbose:
                PETSc.Sys.Print("Output: %s" % count)
            eta.assign(Dn+b)
            Dfile.write(un, Dn, eta, b)
            dumpcount -= dumpfreq
            count += 1
        return dumpcount

    # Initial output dump
    dumpcount = dump(dumpcount, dumpfreq, count)

    # Some diagnostics
    energy = []
    energy_t = assemble(0.5*inner(un, un)*Dn*dx +
                        0.5*g*(Dn + b)*(Dn + b)*dx)
    energy.append(energy_t)
    if verbose:
        PETSc.Sys.Print("Energy: %s" % energy_t)

    t = 0.0

    # Initialise u and D
    u0 = Function(Vu).assign(un)
    D0 = Function(VD).assign(Dn)

    while t < tmax - Dt/2:
        t += Dt

        # First guess for next timestep
        up.assign(un)
        Dp.assign(Dn)

        # Picard cycle
        for i in range(4):

            # Update layer depth
            Dsolver.solve()
            # Update velocity
            Usolver.solve()

            # Calculate increments for up, Dp
            DUsolver.solve()
            PETSc.Sys.Print(
                "Implicit solve finished for Picard iteration %s "
                "at t=%s.\n" % (i + 1, t)
            )

            up += deltau
            Dp += deltaD

        un.assign(up)
        Dn.assign(Dp)

        dumpcount = dump(dumpcount, dumpfreq, count)

        energy_t = assemble(0.5*inner(un, un)*Dn*dx +
                            0.5*g*(Dn + b)*(Dn + b)*dx)
        energy.append(energy_t)
        if verbose:
            PETSc.Sys.Print("Energy: %s" % energy_t)

    UerrL2 = errornorm(un, u0, norm_type="L2")
    DerrL2 = errornorm(Dn, D0, norm_type="L2")

    diffu = Function(Vu).assign(abs(un - u0))
    diffD = Function(VD).assign(abs(Dn - D0))

    # Normalized Linf error
    UerrLinf = diffu.dat.data.max() / u0.dat.data.max()
    DerrLinf = diffD.dat.data.max() / D0.dat.data.max()

    # Normalized L2 error
    UerrL2 = UerrL2/norm(u0, norm_type="L2")
    DerrL2 = DerrL2/norm(D0, norm_type="L2")

    return (UerrL2, DerrL2), (UerrLinf, DerrLinf), mesh


def compute_conv_rates(u):
    """Computes the convergence rate for this particular test case

    :arg u: a list of errors.

    Returns a list of convergence rates. Note the first element of
    the list will be empty, as there is no previous computation to
    compare with. '---' will be inserted into the first component.
    """

    u_array = np.array(u)
    rates = list(np.log2(u_array[:-1] / u_array[1:]))
    rates.insert(0, '---')
    return rates


# Collect errors and mesh information for convergence test
U_L2errs = []
D_L2errs = []
U_Linferrs = []
D_Linferrs = []
num_cells = []
for ref_level in [3, 4, 5, 6, 7]:
    L2errs, Linferrs, mesh = run_williamson2(refinement_level=ref_level,
                                             dumpfreq=args.dumpfreq,
                                             test=args.test,
                                             verbose=args.verbose,
                                             model_degree=args.model_degree,
                                             hybridization=args.hybridization)
    u_errL2, D_errL2 = L2errs
    u_errLinf, D_errLinf = Linferrs
    PETSc.Sys.Print("Normalized L2 error in velocity: %s" % u_errL2)
    PETSc.Sys.Print("Normalized L2 error in depth: %s" % D_errL2)
    PETSc.Sys.Print("Normalized Linf error in velocity: %s" % u_errLinf)
    PETSc.Sys.Print("Normalized Linf error in depth: %s" % D_errLinf)
    U_L2errs.append(u_errL2)
    D_L2errs.append(D_errL2)
    U_Linferrs.append(u_errLinf)
    D_Linferrs.append(D_errLinf)
    num_cells.append(mesh.num_cells())

u_L2rates = compute_conv_rates(U_L2errs)
D_L2rates = compute_conv_rates(D_L2errs)
u_Linfrates = compute_conv_rates(U_Linferrs)
D_Linfrates = compute_conv_rates(D_Linferrs)

data = {"NumCells": num_cells,
        "NormalizedVelocityL2Errors": U_L2errs,
        "NormalizedDepthL2Errors": D_L2errs,
        "VelocityL2Rates": u_L2rates,
        "DepthL2Rates": D_L2rates,
        "NormalizedVelocityLinfErrors": U_Linferrs,
        "NormalizedDepthLinfErrors": D_Linferrs,
        "VelocityLinfRates": u_Linfrates,
        "DepthLinfRates": D_Linfrates}

df = pd.DataFrame(data)
csv_result = "W2-convergence-test"

if args.hybridization:
    csv_result += "-hybridization"

df.to_csv(csv_result + ".csv", index=False, mode="w")
