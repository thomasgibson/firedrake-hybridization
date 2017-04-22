from __future__ import absolute_import, print_function, division

from firedrake import *

from poisson import run_primal_poisson, run_mixed_poisson


for quad in [False, True]:

    primal_u = run_primal_poisson(3, 1, quads=quad)
    sigma_h, u_h, sigma_nh, u_nh, err_s, err_u = run_mixed_poisson(3, 1, quads=quad)

    name_primal = "primal_poisson"
    name_hybrid = "hybrid_poisson"
    name_mixed = "mixed_poisson"

    if quad:
        name_primal += "-quad"
        name_hybrid += "-quad"
        name_mixed += "-quad"

    File(name_primal + ".pvd").write(primal_u)
    File(name_hybrid + ".pvd").write(sigma_h, u_h)
    File(name_mixed + ".pvd").write(sigma_nh, u_nh)

    print(err_s)
    print(err_u)
