from __future__ import absolute_import, division
from firedrake import *
import matplotlib.pyplot as plt
import numpy as np


def poisson_sphere(MeshClass, refinement, hdiv_space, degree):
    """Test hybridizing lowest order mixed methods on a sphere."""
    mesh = MeshClass(refinement_level=refinement)
    mesh.init_cell_orientations(Expression(("x[0]", "x[1]", "x[2]")))
    x, y, z = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, hdiv_space, degree + 1)
    U = FunctionSpace(mesh, "DG", degree)
    W = U * V
    Ue = FunctionSpace(mesh, "CG", degree + 3)
    Ve = FunctionSpace(mesh, hdiv_space, 3*(degree + 1))

    f = Function(U)
    f.interpolate(x*y*z)

    u_exact = Function(Ue).interpolate(x*y*z/12.0)
    sigma_exact = Function(Ve).project(-grad(x*y*z/12.0))

    u, sigma = TrialFunctions(W)
    v, tau = TestFunctions(W)

    a = (dot(sigma, tau) - div(tau)*u + v*div(sigma))*dx
    L = f*v*dx
    w = Function(W)

    nullsp = MixedVectorSpaceBasis(W, [VectorSpaceBasis(constant=True), W[1]])
    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'preonly',
                                'pc_type': 'lu',
                                'pc_factor_mat_solver_package': 'mumps',
                                'hdiv_residual': {'ksp_type': 'cg',
                                                  'ksp_monitor': True,
                                                  'ksp_rtol': 1e-14,
                                                  'pc_type': 'bjacobi',
                                                  'sub_pc_type': 'ilu'},
                                'use_reconstructor': True}}
    solve(a == L, w, nullspace=nullsp, solver_parameters=params)
    u_h, sigma_h = w.split()
    error_u = errornorm(u_exact, u_h)
    error_sigma = errornorm(sigma_exact, sigma_h)
    return error_u, error_sigma


degree = 0
mesh = {"BDM": UnitIcosahedralSphereMesh,
        "RT": UnitIcosahedralSphereMesh,
        "RTCF": UnitCubedSphereMesh}
rterr_u = []
rterr_sigma = []
bdmerr_u = []
bdmerr_sigma = []
rtcferr_u = []
rtcferr_sigma = []
trimesh_cw = []
cubemesh_cw = []
for i in range(2, 8):
    trimesh_cw.append(sqrt((4.0*pi)/mesh["RT"](i).topology.num_cells()))
    cubemesh_cw.append(sqrt((4.0*pi)/mesh["RTCF"](i).topology.num_cells()))
    rt_u_err, rt_s_err = poisson_sphere(mesh["RT"], i, "RT", degree)
    bdm_u_err, bdm_s_err = poisson_sphere(mesh["BDM"], i, "BDM", degree)
    rtcf_u_err, rtcf_s_err = poisson_sphere(mesh["RTCF"], i, "RTCF", degree)

    rterr_u.append(rt_u_err)
    rterr_sigma.append(rt_s_err)
    bdmerr_u.append(bdm_u_err)
    bdmerr_sigma.append(bdm_s_err)
    rtcferr_u.append(rtcf_u_err)
    rtcferr_sigma.append(rtcf_s_err)

rterr_u = np.asarray(rterr_u)
rterr_sigma = np.asarray(rterr_sigma)
bdmerr_u = np.asarray(bdmerr_u)
bdmerr_sigma = np.asarray(bdmerr_sigma)
rtcferr_u = np.asarray(rtcferr_u)
rtcferr_sigma = np.asarray(rtcferr_sigma)

fig = plt.figure()
ax = fig.add_subplot(111)

dh = np.asarray(cubemesh_cw)
k = degree + 1
dh_arry = dh ** k
dh_arry = 0.001 * dh_arry

orange = '#FF6600'
lw = '5'
ms = 15

if k == 1:
    dhlabel = '$\propto \Delta h$'
else:
    dhlabel = '$\propto \Delta h^%d$' % k

print "RT EOC for p: %f" % np.log2(rterr_u[:-1]/rterr_u[1:])[-1]
print "BDM EOC for p: %f" % np.log2(bdmerr_u[:-1]/bdmerr_u[1:])[-1]
print "RTCF EOC for p: %f" % np.log2(rtcferr_u[:-1]/rtcferr_u[1:])[-1]

print "RT EOC for u: %f" % np.log2(rterr_sigma[:-1]/rterr_sigma[1:])[-1]
print "BDM EOC for u: %f" % np.log2(bdmerr_sigma[:-1]/bdmerr_sigma[1:])[-1]
print "RTCF EOC for u: %f" % np.log2(rtcferr_sigma[:-1]/rtcferr_sigma[1:])[-1]

ax.loglog(trimesh_cw, rterr_u, color='r', marker='o',
          linestyle='-', linewidth=lw, markersize=ms,
          label='$RT_1$ $DG_0$ $p_h$')
ax.loglog(trimesh_cw, rterr_sigma, color='b', marker='^',
          linestyle='-', linewidth=lw, markersize=ms,
          label='$RT_1$ $DG_0$ $u_h$')
ax.loglog(trimesh_cw, bdmerr_u, color='c', marker='o',
          linestyle='--', linewidth=lw, markersize=ms,
          label='$BDM_1$ $DG_0$ $p_h$')
ax.loglog(trimesh_cw, bdmerr_sigma, color=orange, marker='^',
          linestyle='--', linewidth=lw, markersize=ms,
          label='$BDM_1$ $DG_0$ $u_h$')
ax.loglog(cubemesh_cw, rtcferr_u, color='g', marker='o',
          linewidth=lw, markersize=ms,
          label='$RTCF_1$ $DQ_0$ $p_h$')
ax.loglog(cubemesh_cw, rtcferr_sigma, color='m', marker='^',
          linewidth=lw, markersize=ms,
          label='$RTCF_1$ $DQ_0$ $u_h$')
ax.loglog(trimesh_cw, dh_arry, color='k', linestyle=':',
          linewidth=lw, label=dhlabel)
ax.grid(True)
plt.title("Resolution test for LO H-RT, H-BDM, and H-RTCF methods")
plt.xlabel("Cell width $\Delta h$")
plt.ylabel("$L^2$-error against projected exact solution")
plt.gca().invert_xaxis()
plt.legend(loc=2)
font = {'family': 'normal',
        'weight': 'bold',
        'size': 28}
plt.rc('font', **font)
plt.gca().invert_xaxis()
plt.show()
