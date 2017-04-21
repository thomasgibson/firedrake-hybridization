from __future__ import absolute_import, print_function, division

from firedrake import *

from helmholtz import run_helmholtz
from poisson import run_primal_poisson


for flag in [False, True]:
    vel, pr = run_helmholtz(3, 1, quads=flag)

    if flag:
        name = "2D-helmholtz-quad"
    else:
        name = "2D-helmholtz"

    File(name + ".pvd").write(vel, pr)

for flag in [False, True]:
    u = run_primal_poisson(3, 1, quads=flag)

    if flag:
        name = "3D-primalpoisson-quad"
    else:
        name = "3D-primalpoisson"

    File(name + ".pvd").write(u)
