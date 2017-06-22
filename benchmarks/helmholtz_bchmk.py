from __future__ import absolute_import, division
from hybrid_solvers import *
from firedrake import *
import numpy as np


errs = []
for i in range(2, 8):
    problem = HybridMixedHelmholtzProblem(degree=1, resolution=i, hexes=False)
    solver = HybridSolver(problem, post_process=True)
    solver.solve()
    u = solver.computed_solution
    exact = problem.exact_solution
    err = errornorm(u, exact)
    errs.append(err)

errs = np.array(errs)
print np.log2(errs[:-1] / errs[1:])[-1]
