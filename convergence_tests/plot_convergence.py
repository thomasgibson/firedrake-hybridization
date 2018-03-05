from firedrake import *
import os
import sys
import pandas as pd
import seaborn
import matplotlib

from matplotlib import pyplot as plt
from mpltools import annotation


FONTSIZE = 14
MARKERSIZE = 10
LINEWIDTH = 3

data = "W2-convergence-test-hybridization.csv"
if not os.path.exists(data):
    print("Cannot find data file '%s'" % data)
    sys.exit(1)

df = pd.read_csv(data)

seaborn.set(style="ticks")

# Set up mesh information
refs = [3, 4, 5]

num_cells = []
u_dofs = []
D_dofs = []
avg_mesh_size = []
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
    u_dofs.append(u.dof_dset.layout_vec.getSize())
    D_dofs.append(D.dof_dset.layout_vec.getSize())

colors = ['#30a2da', '#fc4f30']
markers = iter(["o", "s", "^", "D"])
linestyles = iter(["solid", "dashed", "dashdot", "dotted"])

fig, (axes,) = plt.subplots(1, 1, figsize=(6, 4), squeeze=False)
ax, = axes
ax.set_ylabel("Normalized error", fontsize=FONTSIZE+2)

ax.spines["left"].set_position(("outward", 10))
ax.spines["bottom"].set_position(("outward", 10))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
ax.set_xscale('log')
ax.set_yscale('log')
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax.plot(avg_mesh_size, df.NormalizedDepthL2Errors,
        label="$L^2(D)$",
        linewidth=LINEWIDTH,
        linestyle=next(linestyles),
        markersize=MARKERSIZE,
        marker=next(markers),
        color=colors[0],
        clip_on=False)

ax.plot(avg_mesh_size, df.NormalizedVelocityL2Errors,
        label="$L^2(u)$",
        linewidth=LINEWIDTH,
        linestyle=next(linestyles),
        markersize=MARKERSIZE,
        marker=next(markers),
        color=colors[1],
        clip_on=False)

annotation.slope_marker((2e5, 5.e-4), 2, ax=ax,
                        invert=True,
                        poly_kwargs={'facecolor': colors[0]})

for tick in ax.get_xticklabels():
    tick.set_fontsize(FONTSIZE)

for tick in ax.get_yticklabels():
    tick.set_fontsize(FONTSIZE)

xlabel = fig.text(0.5, -0.1,
                  "Average mesh size $\Delta x$",
                  ha='center',
                  fontsize=FONTSIZE+2)


def update_xlabels(ax):
    xlabels = [format(label, '.0E') for label in ax.get_xticks()]
    ax.set_xticklabels(xlabels)

update_xlabels(ax)

handles, labels = ax.get_legend_handles_labels()
legend = fig.legend(handles, labels,
                    loc=9,
                    bbox_to_anchor=(0.5, 1.175),
                    bbox_transform=fig.transFigure,
                    ncol=2,
                    handlelength=2,
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
