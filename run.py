from __future__ import absolute_import, print_function, division

from firedrake import *

from helmholtz import run_helmholtz
from poisson import run_primal_poisson, run_hybrid_poisson


for flag in [False, True]:
    u = run_primal_poisson(4, 1, quads=flag)
    sigmah, uh = run_hybrid_poisson(4, 2, quads=flag)
    vel, pr = run_helmholtz(4, 1, quads=flag)

    if flag:
        name = "3D-primalpoisson-quad"
        name2 = "3D-hybridpoisson-quad"
        name3 = "2D-helmholtz-quad"
    else:
        name = "3D-primalpoisson"
        name2 = "3D-hybridpoisson"
        name3 = "2D-helmholtz"

    File(name + ".pvd").write(u)
    File(name2 + ".pvd").write(sigmah, uh)
    File(name3 + ".pvd").write(vel, pr)
