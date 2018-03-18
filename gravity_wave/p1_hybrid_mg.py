from firedrake import *

import sys
import petsc4py

petsc4py.init(sys.argv)

from petsc4py import PETSc


__all__ = ["P1HMultiGrid"]


class P1HMultiGrid(object):
    """
    """

    def __init__(self, S, fine_solution, omega_c2=1):
        """
        """

        self.lambda_f = fine_solution
        self.trace_operator = S
        self._WT = self.lambda_f.function_space()
        self._trace_r = Function(self._WT)

        self._P1 = FunctionSpace(self._WT.mesh(), "CG", 1)

        # Need an index set to copy data into PETSc vecs
        with self._trace_r.dat.vec as v:
            ndof = self._WT.dof_dset.size
            self._idx_set = PETSc.IS().createStride(ndof,
                                                    first=v.owner_range[0],
                                                    step=1,
                                                    comm=v.comm)

        # Linear solver for prolongation to trace
        gammar = TestFunction(self._WT)
        lambdar = TrialFunction(self._WT)
        M_t = assemble(gammar('+')*lambdar('+')*(dS_v + dS_h)
                       + gammar*lambdar*ds_tb)

        tparams = {'ksp_type': 'cg',
                   'pc_type': 'bjacobi',
                   'sub_pc_type': 'ilu'}
        self._trace_mass_solve = LinearSolver(M_t, solver_parameters=tparams)

        # Pre-smoothing with a 4-point Jacobi method
        # accelerated by Chebyshev
        smootherparams = {'ksp_type': 'chebyshev',
                          'ksp_max_it': 4,
                          'ksp_convergence_test': 'skip',
                          'ksp_initial_guess_nonzero': True,
                          'pc_type': 'bjacobi',
                          'sub_pc_type': 'ilu'}
        self._presmoother = LinearSolver(self.trace_operator,
                                         solver_parameters=smootherparams)

        # Coarse grid correction problem (positive-definite Helmholtz)
        u = TrialFunction(self._P1)
        v = TestFunction(self._P1)
        L = assemble(-(inner(v, u)*dx +
                       omega_c2*dot(grad(v), grad(u))*dx))

        # Firedrake GMG is broken for this problem...
        # cparams = {"ksp_type": "preonly",
        #            "pc_type": "mg",
        #            "pc_mg_type": "full",
        #            "mg_levels_ksp_type": "chebyshev",
        #            "mg_levels_ksp_max_it": 2,
        #            "mg_levels_pc_type": "bjacobi",
        #            "mg_levels_sub_pc_type": "ilu"}
        cparams = {'ksp_type': 'preonly',
                   'pc_type': 'hypre',
                   'pc_hypre_type': 'boomeramg',
                   'pc_hypre_boomeramg_no_CF': False,
                   'pc_hypre_boomeramg_coarsen_type': 'Falgout',
                   'pc_hypre_boomeramg_interp_type': 'classical',
                   'pc_hypre_boomeramg_P_max': 4,
                   'pc_hypre_boomeramg_agg_nl': 1,
                   'pc_hypre_boomeramg_max_level': 3,
                   'pc_hypre_boomeramg_strong_threshold': 0.25}
        self._coarse_grid_solver = LinearSolver(L, solver_parameters=cparams)

        # Post-smoothing with a 5-point bJacobi method
        # accelerated with Chebyshev
        postsmoothparams = {'ksp_type': 'chebyshev',
                            'ksp_max_it': 5,
                            'ksp_convergence_test': 'skip',
                            'ksp_initial_guess_nonzero': True,
                            'pc_type': 'bjacobi',
                            'sub_pc_type': 'ilu'}
        self._postsmoother = LinearSolver(self.trace_operator,
                                          solver_parameters=postsmoothparams)

    def restrict_to_p1(self, r_lambda, r_coarse):
        """
        """

        temp = Function(self._WT)
        self._trace_mass_solve.solve(temp, r_lambda)
        v = TestFunction(self._P1)
        r_coarse.assign(assemble(v('+')*temp('+')*(dS_v + dS_h) +
                                 v*temp*ds_tb))

    def prolong_to_trace(self, du, lambdar):
        """
        """

        gammar = TestFunction(self._WT)
        rhs = assemble(gammar('+')*du('+')*(dS_v + dS_h) +
                       gammar*du*ds_tb)
        self._trace_mass_solve.solve(lambdar, rhs)

    def solve(self, lambdar, r):
        """
        """

        # RHS for the coarse problem
        r_coarse = Function(self._P1)

        # coarse-grid solution
        du = Function(self._P1)

        # prolongated update to lambda
        dlambdar = Function(self._WT)

        # Algorithm:
        # 1. Pre-smooth
        self._presmoother.solve(lambdar, r)

        # Need to restrict the residual, but all
        # we have are assembled matrices. So we
        # manually perform the matvec to get the
        # residual.
        residual = Function(self._WT)
        with lambdar.dat.vec as v:
            with residual.dat.vec as q:
                q -= self.trace_operator._M.handle*v

        # 2. Restrict the residual to P1
        self.restrict_to_p1(residual, r_coarse)

        # 3. Coarse grid solve
        du.assign(0.0)
        self._coarse_grid_solver.solve(du, r_coarse)

        # 4. Prolongate coarse grid update to trace space
        self.prolong_to_trace(du, dlambdar)
        lambdar += dlambdar

        # 5. Postsmooth the update
        self._postsmoother.solve(lambdar, r)

    def apply(self, pc, x, y):
        """
        """

        with self._trace_r.dat.vec as v:
            temp = x.getSubVector(self._idx_set)
            x.copy(v)
            x.restoreSubVector(self._idx_set, temp)

        self.solve(self.lambda_f, self._trace_r)

        with self.lambda_f.dat.vec_ro as v:
            v.copy(y)
