from __future__ import absolute_import, division

from firedrake import *
import numpy as np
import matplotlib.pyplot as plt


def run_hybrid_extr_helmholtz(degree, res, quads=False, write=False):
    nx = 2 ** res
    ny = 2 ** res
    nz = 2 ** res
    h = 0.2 / nz
    base = UnitSquareMesh(nx, ny, quadrilateral=quads)
    mesh = ExtrudedMesh(base, layers=nz, layer_height=h)

    if quads:
        RT = FiniteElement("RTCF", quadrilateral, degree + 1)
        DG_v = FiniteElement("DG", interval, degree)
        DG_h = FiniteElement("DQ", quadrilateral, degree)
        CG = FiniteElement("CG", interval, degree + 1)

    else:
        RT = FiniteElement("RT", triangle, degree + 1)
        DG_v = FiniteElement("DG", interval, degree)
        DG_h = FiniteElement("DG", triangle, degree)
        CG = FiniteElement("CG", interval, degree + 1)

    HDiv_ele = EnrichedElement(HDiv(TensorProductElement(RT, DG_v)),
                               HDiv(TensorProductElement(DG_h, CG)))
    V = FunctionSpace(mesh, HDiv_ele)
    U = FunctionSpace(mesh, "DG", degree)
    W = V * U

    x = SpatialCoordinate(mesh)
    f = Function(U)
    f.interpolate((1+38*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*3)*sin(x[2]*pi*5))
    exact = Function(U)
    exact.interpolate(sin(x[0]*pi*2)*sin(x[1]*pi*3)*sin(x[2]*pi*5))
    exact.rename("exact")
    exact_flux = Function(V)
    exact_flux.project(-grad(sin(x[0]*pi*2)*sin(x[1]*pi*3)*sin(x[2]*pi*5)))
    exact_flux.rename("exact_flux")

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    a = dot(sigma, tau)*dx + u*v*dx + div(sigma)*v*dx - div(tau)*u*dx
    L = f*v*dx
    w = Function(W)
    params = {'ksp_type': 'preonly',
              'mat_type': 'matfree',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'cg',
                                'pc_type': 'hypre',
                                'pc_hypre_type': 'boomeramg',
                                'ksp_rtol': 1e-8,
                                'ksp_monitor': True,
                                'hdiv_residual': {'ksp_type': 'cg',
                                                  'pc_type': 'bjacobi',
                                                  'sub_pc_type': 'ilu',
                                                  'ksp_rtol': 1e-8,
                                                  'ksp_monitor': True},
                                'use_reconstructor': True}}
    solve(a == L, w, solver_parameters=params)
    sigma_h, u_h = w.split()
    sigma_h.rename("flux")
    u_h.rename("pressure")

    err_s = errornorm(u_h, exact)
    err_f = errornorm(sigma_h, exact_flux)

    if write:
        File("3D-hybrid.pvd").write(sigma_h, u_h, exact, exact_flux)
        return
    else:
        return (err_s, err_f), mesh

ref_levels = range(2, 7)
degree = 0
errRT_u = []
errRT_sigma = []
errRTCF_u = []
errRTCF_sigma = []
mRT = []
mRTCF = []
for i in ref_levels:
    e, mrt = run_hybrid_extr_helmholtz(degree=degree,
                                       res=i, quads=False)
    rt_err_s, rt_err_f = e
    ertc, mrtcf = run_hybrid_extr_helmholtz(degree=degree,
                                            res=i, quads=True)
    rtcf_err_s, rtcf_err_f = ertc

    mRT.append(mrt)
    mRTCF.append(mrtcf)
    errRT_u.append(rt_err_s)
    errRT_sigma.append(rt_err_f)
    errRTCF_u.append(rtcf_err_s)
    errRTCF_sigma.append(rtcf_err_f)

errRT_u = np.asarray(errRT_u)
errRT_sigma = np.asarray(errRT_sigma)
errRTCF_u = np.asarray(errRTCF_u)
errRTCF_sigma = np.asarray(errRTCF_sigma)
rtdof = mRT[-1].topology.num_cells()*6
rtcfdof = mRTCF[-1].topology.num_cells()*7

print "RT Dof count: %d" % rtdof
print "RT perr: %f" % errRT_u[-1]
print "RT uerr: %f" % errRT_sigma[-1]
print "RT p EOC: %f" % np.log2(errRT_u[:-1]/errRT_u[1:])[-1]
print "RT u EOC: %f" % np.log2(errRT_sigma[:-1]/errRT_sigma[1:])[-1]

print "RTCF Dof count: %d" % rtcfdof
print "RTCF perr %f" % errRTCF_u[-1]
print "RTCF uerr %f" % errRTCF_sigma[-1]
print "RTCF p EOC: %f" % np.log2(errRTCF_u[:-1]/errRTCF_u[1:])[-1]
print "RTCF u EOC: %f" % np.log2(errRTCF_sigma[:-1]/errRTCF_sigma[1:])[-1]

fig = plt.figure()
ax = fig.add_subplot(111)

res = [2 ** r for r in ref_levels]
dh = np.asarray(res)
dh_arry = dh ** 2
dh_arry = 0.0001 * dh_arry

orange = '#FF6600'
lw = '5'
ms = 15

ax.loglog(res, errRT_u, color='r', marker='o',
          linestyle='-', linewidth=lw, markersize=ms,
          label='$DG_0$ $p_h$')
ax.loglog(res, errRT_sigma, color='b', marker='^',
          linestyle='-', linewidth=lw, markersize=ms,
          label='$RT_1$ $u_h$')
ax.loglog(res, errRTCF_u, color='c', marker='o',
          linestyle='-', linewidth=lw, markersize=ms,
          label='$DQ_0$ $p_h$')
ax.loglog(res, errRTCF_sigma, color=orange, marker='^',
          linestyle='-', linewidth=lw, markersize=ms,
          label='$RTCF_1$ $u_h$')
ax.loglog(res, dh_arry[::-1], color='k', linestyle=':',
          linewidth=lw, label='$\propto \Delta x^2$')
ax.grid(True)
plt.title("Resolution test for lowest order H-RT and H-RTCF methods")
plt.xlabel("Mesh resolution in all spatial directions $2^r$")
plt.ylabel("$L^2$-error against projected exact solution")
plt.gca().invert_xaxis()
plt.legend(loc=1)
font = {'family': 'normal',
        'weight': 'bold',
        'size': 28}
plt.rc('font', **font)
plt.show()
