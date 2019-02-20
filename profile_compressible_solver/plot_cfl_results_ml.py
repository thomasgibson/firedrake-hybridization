import os
import pandas as pd
import seaborn

from matplotlib import pyplot as plt


FONTSIZE = 14
MARKERSIZE = 10
LINEWIDTH = 3

cfl_range = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

rt0_ml_data = ["results/RT0_dx346km_dz156m_cfl%s_Hybrid_SCPC_fgmres_ml_richardson_data.csv"
               % cfl for cfl in cfl_range]
rt0_ml_str_data = ["results/RT0_dx346km_dz156m_cfl%s_Hybrid_SCPC_fgmres_ml_richardson_stronger_data.csv"
                   % cfl for cfl in cfl_range]
rt1_ml_data = ["results/RT1_dx346km_dz156m_cfl%s_Hybrid_SCPC_fgmres_ml_richardson_data.csv"
               % cfl for cfl in cfl_range]
rt1_ml_str_data = ["results/RT1_dx346km_dz156m_cfl%s_Hybrid_SCPC_fgmres_ml_richardson_stronger_data.csv"
                   % cfl for cfl in cfl_range]
bdfm1_ml_data = ["results/BDFM1_dx346km_dz156m_cfl%s_Hybrid_SCPC_fgmres_ml_richardson_data.csv"
                 % cfl for cfl in cfl_range]
bdfm1_ml_str_data = ["results/BDFM1_dx346km_dz156m_cfl%s_Hybrid_SCPC_fgmres_ml_richardson_stronger_data.csv"
                     % cfl for cfl in cfl_range]


for data in (rt0_ml_data + rt1_ml_data +
             rt0_ml_str_data + rt1_ml_str_data +
             bdfm1_ml_data + bdfm1_ml_str_data):
    if not os.path.exists(data):
        import sys
        print("Cannot find data file: %s" % data)
        sys.exit(1)


fig, (axes,) = plt.subplots(1, 1, figsize=(7, 5), squeeze=False)
ax, = axes
ax.set_ylim(0, 35)
ax.spines["left"].set_position(("outward", 10))
ax.spines["bottom"].set_position(("outward", 10))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
ax.set_xticks(cfl_range)
ax.set_ylabel("Krylov iterations", fontsize=FONTSIZE+2)

# Create data matrices
rt0_ml_dfs = pd.concat(pd.read_csv(d) for d in rt0_ml_data)
rt0_ml_str_dfs = pd.concat(pd.read_csv(d) for d in rt0_ml_str_data)
rt1_ml_dfs = pd.concat(pd.read_csv(d) for d in rt1_ml_data)
rt1_ml_str_dfs = pd.concat(pd.read_csv(d) for d in rt1_ml_str_data)
bdfm1_ml_dfs = pd.concat(pd.read_csv(d) for d in bdfm1_ml_data)
bdfm1_ml_str_dfs = pd.concat(pd.read_csv(d) for d in bdfm1_ml_str_data)

# Organize and group by CFL number
rt0_ml_grps = rt0_ml_dfs.groupby(["horizontal_courant"],
                                 as_index=False)
rt1_ml_grps = rt1_ml_dfs.groupby(["horizontal_courant"],
                                 as_index=False)
rt0_ml_str_grps = rt0_ml_str_dfs.groupby(["horizontal_courant"],
                                         as_index=False)
rt1_ml_str_grps = rt1_ml_str_dfs.groupby(["horizontal_courant"],
                                         as_index=False)
bdfm1_ml_grps = bdfm1_ml_dfs.groupby(["horizontal_courant"],
                                     as_index=False)
bdfm1_ml_str_grps = bdfm1_ml_str_dfs.groupby(["horizontal_courant"],
                                             as_index=False)

# RT0, fgmres + ML(richardson(3))
rt0_ml_cfls = []
rt0_ml_iters = []
for group in rt0_ml_grps:

    cfl, df = group
    rt0_ml_cfls.append(cfl)
    rt0_ml_iters.append(df.ksp_iters)

# RT1, fgmres + ML(richardson(3))
rt1_ml_cfls = []
rt1_ml_iters = []
for group in rt1_ml_grps:

    cfl, df = group
    rt1_ml_cfls.append(cfl)
    rt1_ml_iters.append(df.ksp_iters)

# RT0, fgmres + ML(richardson(5))
rt0_ml_str_cfls = []
rt0_ml_str_iters = []
for group in rt0_ml_str_grps:

    cfl, df = group
    rt0_ml_str_cfls.append(cfl)
    rt0_ml_str_iters.append(df.ksp_iters)

# RT1, fgmres + ML(richardson(5))
rt1_ml_str_cfls = []
rt1_ml_str_iters = []
for group in rt1_ml_str_grps:

    cfl, df = group
    rt1_ml_str_cfls.append(cfl)
    rt1_ml_str_iters.append(df.ksp_iters)

# BDFM1, fgmres + ML(richardson(3))
bdfm1_ml_cfls = []
bdfm1_ml_iters = []
for group in bdfm1_ml_grps:

    cfl, df = group
    bdfm1_ml_cfls.append(cfl)
    bdfm1_ml_iters.append(df.ksp_iters)

# BDFM1, fgmres + ML(richardson(5))
bdfm1_ml_str_cfls = []
bdfm1_ml_str_iters = []
for group in bdfm1_ml_str_grps:

    cfl, df = group
    bdfm1_ml_str_cfls.append(cfl)
    bdfm1_ml_str_iters.append(df.ksp_iters)

ax.plot(rt0_ml_cfls, rt0_ml_iters,
        label="$RT_0$ $ml(\\mathcal{R}(3))$",
        markersize=MARKERSIZE,
        linewidth=LINEWIDTH,
        color="k",
        marker="o",
        linestyle="solid")

ax.plot(rt0_ml_str_cfls, rt0_ml_str_iters,
        label="$RT_0$ $ml(\\mathcal{R}(5))$",
        markersize=MARKERSIZE,
        linewidth=LINEWIDTH,
        color="k",
        marker="o",
        linestyle="dotted")

ax.plot(rt1_ml_cfls, rt1_ml_iters,
        label="$RT_1$ $ml(\\mathcal{R}(3))$",
        markersize=MARKERSIZE,
        linewidth=LINEWIDTH,
        color="m",
        marker="^",
        linestyle="solid")

ax.plot(rt1_ml_str_cfls, rt1_ml_str_iters,
        label="$RT_1$ $ml(\\mathcal{R}(5))$",
        markersize=MARKERSIZE,
        linewidth=LINEWIDTH,
        color="m",
        marker="^",
        linestyle="dotted")

ax.plot(bdfm1_ml_cfls, bdfm1_ml_iters,
        label="$BDFM_1$ $ml(\\mathcal{R}(3))$",
        markersize=MARKERSIZE,
        linewidth=LINEWIDTH,
        color="g",
        marker="*",
        linestyle="solid")

ax.plot(bdfm1_ml_str_cfls, bdfm1_ml_str_iters,
        label="$BDFM_1$ $ml(\\mathcal{R}(5))$",
        markersize=MARKERSIZE,
        linewidth=LINEWIDTH,
        color="g",
        marker="*",
        linestyle="dotted")


for tick in ax.get_xticklabels():
    tick.set_fontsize(FONTSIZE)

for tick in ax.get_yticklabels():
    tick.set_fontsize(FONTSIZE)

ax.grid(b=True, which='major', linestyle='-.')

xlabel = fig.text(0.5, -0.15,
                  "Horiz. CFL number\n $\\sqrt{\\frac{c_p T_0}{\\gamma}}\\frac{\\Delta t}{\\Delta x}$",
                  ha='center',
                  fontsize=FONTSIZE)

handles, labels = ax.get_legend_handles_labels()
legend = fig.legend(handles, labels,
                    loc=9,
                    bbox_to_anchor=(0.5, 1.1),
                    bbox_transform=fig.transFigure,
                    ncol=3,
                    handlelength=1.5,
                    fontsize=FONTSIZE-2,
                    numpoints=1,
                    frameon=False)

fig.savefig("cfl_vs_iter_ml.pdf",
            orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight",
            bbox_extra_artists=[xlabel, legend])
