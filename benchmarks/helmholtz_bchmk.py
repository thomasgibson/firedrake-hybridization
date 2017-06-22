from __future__ import absolute_import, division
from hybrid_solvers import *
from firedrake import *


problem = HybridMixedHelmholtzProblem(degree=1, resolution=8, hexes=False)
solver = HybridSolver(problem)
solver.solve()
sigma, u = solver.computed_solution
exact = problem.exact_solution
sigma.rename("velocity")
u.rename("pressure")

File("test.pvd").write(u, sigma, exact)
