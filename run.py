from __future__ import absolute_import, print_function, division

from firedrake import *

from helmholtz import run_helmholtz


for flag in [False, True]:
    vel, pr = run_helmholtz(3, 1, quads=flag)

    if flag:
        name = "2D-helmholtz-quad"
    else:
        name = "2D-helmholtz"

    File(name + ".pvd").write(vel, pr)
