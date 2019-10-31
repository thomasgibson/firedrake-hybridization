from firedrake import *
import os
import sys
import pandas as pd
import seaborn
import matplotlib
import numpy as np

from matplotlib import pyplot as plt

FONTSIZE = 16
MARKERSIZE = 10
LINEWIDTH = 2

data = "W2-convergence-test-hybridization.csv"
if not os.path.exists(data):
    print("Cannot find data file '%s'" % data)
    sys.exit(1)

df = pd.read_csv(data)

# Set up mesh information
refs = [3, 4, 5, 6]

num_cells = []
u_dofs = []
D_dofs = []
avg_mesh_size = []
dx2 = []
for ref in refs:
    R = 6371220.
    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=ref,
                                 degree=3)

    x = SpatialCoordinate(mesh)
    global_normal = as_vector(x)
    mesh.init_cell_orientations(global_normal)

    V1 = FunctionSpace(mesh, "BDM", 2)
    V2 = FunctionSpace(mesh, "DG", 1)
    DG = FunctionSpace(mesh, "DG", 0)

    u = Function(V1)
    D = Function(V2)
    h = Function(DG)
    h.interpolate(CellVolume(mesh))
    mean = h.dat.data.mean()
    cells = mesh.num_cells()

    a = 4*pi*R**2
    num_cells.append(cells)
    avg_mesh_size.append(sqrt(a/cells))
    dx2.append(4*pi/cells)
    u_dofs.append(u.dof_dset.layout_vec.getSize())
    D_dofs.append(D.dof_dset.layout_vec.getSize())

dx2 = [x*0.5e-1 for x in dx2]

colors = ['#30a2da', '#fc4f30']
markers = iter(["o", "s", "^", "D"])
linestyles = iter(["solid", "dashed", "dashdot", "dotted"])

plt.style.use("seaborn-darkgrid")

fig, (axes,) = plt.subplots(1, 1, figsize=(6, 5), squeeze=False)
ax, = axes
ax.set_ylabel("Normalized $L^2$ error", fontsize=FONTSIZE)

ax.spines["left"].set_position(("outward", 10))
ax.spines["bottom"].set_position(("outward", 10))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(6e4, 1e6)
ax.grid(b=True, which='major', linestyle='-.')

derrs = np.array(df.NormalizedDepthL2Errors)
hh = np.array(avg_mesh_size)
convrates_D = np.log(derrs[:-1] / derrs[1:])/np.log(hh[:-1] / hh[1:])
print(convrates_D)

uerrs = np.array(df.NormalizedVelocityL2Errors)
convrates_u = np.log(uerrs[:-1] / uerrs[1:])/np.log(hh[:-1] / hh[1:])
print(convrates_u)

ax.plot(avg_mesh_size, df.NormalizedDepthL2Errors,
        label="$L_{err}^2(D)$",
        linewidth=LINEWIDTH,
        linestyle='solid',
        markersize=MARKERSIZE,
        marker=next(markers),
        color=colors[0],
        clip_on=False)

ax.plot(avg_mesh_size, df.NormalizedVelocityL2Errors,
        label="$L_{err}^2(u)$",
        linewidth=LINEWIDTH,
        linestyle='solid',
        markersize=MARKERSIZE,
        marker=next(markers),
        color=colors[1],
        clip_on=False)

ax.plot(avg_mesh_size, dx2,
        label="$\propto \Delta x^2$",
        linewidth=LINEWIDTH,
        linestyle='solid',
        marker=None,
        color='k',
        clip_on=False)


xlabel = fig.text(0.5, -.05,
                  "Average mesh size $\Delta x$",
                  ha='center',
                  fontsize=FONTSIZE)

handles, labels = ax.get_legend_handles_labels()
legend = fig.legend(handles, labels,
                    loc=9,
                    bbox_to_anchor=(0.5, 1),
                    bbox_transform=fig.transFigure,
                    ncol=3,
                    handlelength=3,
                    fontsize=FONTSIZE,
                    numpoints=1,
                    frameon=False)

ax.tick_params(axis='both', which='major', labelsize=FONTSIZE-2)

# ax.set_xticks(hh)
# ax.set_xticklabels(['{:5.1E}'.format(x) for x in hh])
# seaborn.despine(fig)
fig.savefig("swe-convergence.pdf",
            orientation="landscape",
            format="pdf",
            bbox_inches="tight",
            bbox_extra_artists=[xlabel, legend])
