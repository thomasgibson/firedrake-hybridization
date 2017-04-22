from __future__ import absolute_import, print_function, division

from firedrake import *

from poisson import run_primal_poisson, run_mixed_poisson

params = {"mat_type": "matfree",
          "pc_type": "python",
          "pc_python_type": "firedrake.HybridizationPC",
          "hybridization_pc_type": "hypre",
          "hybridization_pc_hypre_type": "boomeramg",
          "hybridization_ksp_type": "preonly",
          "hybridization_ksp_rtol": 1e-14}

params2 = {"pc_type": "fieldsplit",
           "pc_fieldsplit_type": "schur",
           "ksp_type": "gmres",
           "pc_fieldsplit_schur_fact_type": "FULL",
           "fieldsplit_0_ksp_type": "cg",
           "fieldsplit_0_pc_factor_shift_type": "INBLOCKS",
           "fieldsplit_1_pc_factor_shift_type": "INBLOCKS",
           "fieldsplit_1_ksp_type": "cg"}


for quad in [False, True]:

    primal_u = run_primal_poisson(3, 1, quads=quad)
    sigma_h, u_h = run_mixed_poisson(3, 1, quads=quad, params=params)
    sigma, u = run_mixed_poisson(3, 1, quads=quad, params=params2)

    name_primal = "primal_poisson"
    name_hybrid = "hybrid_poisson"
    name_mixed = "mixed_poisson"

    if quad:
        name_primal += "-quad"
        name_hybrid += "-quad"
        name_mixed += "-quad"

    File(name_primal + ".pvd").write(primal_u)
    File(name_hybrid + ".pvd").write(sigma_h, u_h)
    File(name_mixed + ".pvd").write(sigma, u)
