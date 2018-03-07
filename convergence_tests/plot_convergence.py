from firedrake import *
import os
import sys
import pandas as pd
import seaborn
import matplotlib

from matplotlib import pyplot as plt


FONTSIZE = 16
MARKERSIZE = 10
LINEWIDTH = 3

data = "W2-convergence-test-hybridization.csv"
if not os.path.exists(data):
    print("Cannot find data file '%s'" % data)
    sys.exit(1)

df = pd.read_csv(data)

seaborn.set(style="ticks")

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

    global_normal = Expression(("x[0]", "x[1]", "x[2]"))
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

colors = ['#30a2da', '#fc4f30']
markers = iter(["o", "s", "^", "D"])
linestyles = iter(["solid", "dashed", "dashdot", "dotted"])

fig, (axes,) = plt.subplots(1, 1, figsize=(6, 5), squeeze=False)
ax, = axes
ax.set_ylabel("Normalized error", fontsize=FONTSIZE)

ax.spines["left"].set_position(("outward", 10))
ax.spines["bottom"].set_position(("outward", 10))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
ax.set_xscale('log')
ax.set_yscale('log')

ax.plot(avg_mesh_size, df.NormalizedDepthL2Errors,
        label="$L^2(D)$",
        linewidth=LINEWIDTH,
        linestyle='solid',
        markersize=MARKERSIZE,
        marker=next(markers),
        color=colors[0],
        clip_on=False)

ax.plot(avg_mesh_size, df.NormalizedVelocityL2Errors,
        label="$L^2(u)$",
        linewidth=LINEWIDTH,
        linestyle='dashed',
        markersize=MARKERSIZE,
        marker=next(markers),
        color=colors[1],
        clip_on=False)

ax.plot(avg_mesh_size, dx2,
        label="$\propto \Delta x^2$",
        linewidth=LINEWIDTH,
        linestyle='dotted',
        marker=None,
        color='k',
        clip_on=False)


xlabel = fig.text(0.5, -.05,
                  "Average mesh size $\Delta h$",
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

seaborn.despine(fig)
fig.savefig("swe-convergence.pdf",
            orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight",
            bbox_extra_artists=[xlabel, legend])
