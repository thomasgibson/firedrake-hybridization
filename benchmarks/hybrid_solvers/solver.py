from __future__ import absolute_import, division
from firedrake import *


class HybridSolver(object):
    """
    """

    def __init__(self, problem, params=None, post_process=False):
        """
        """
        self.problem = problem
        if params is None:
            self.params = {"ksp_type": "preonly",
                           "pc_type": "lu",
                           "pc_factor_mat_solver_package": "mumps"}
        else:
            self.params = params
        self.post_process = post_process
        self.broken_solution = problem.broken_w
        self.conforming_solution = problem.w
        self.lambdar_sol = problem.lambdar_sol
        self._solved = False
        self._setup()

    def _setup(self):
        """
        """
        from firedrake.assemble import (allocate_matrix,
                                        create_assembly_callable)
        from firedrake.formmanipulation import split_form

        M = Tensor(self.problem.a)
        K = Tensor(self.problem.multiplier_integrals)
        F = Tensor(self.problem.L)

        schur_comp = K * M.inv * K.T
        self.S = allocate_matrix(schur_comp, bcs=self.problem.trace_bcs)
        self.E = assemble(K * M.inv * F)
        self._assemble_S = create_assembly_callable(schur_comp,
                                                    tensor=self.S,
                                                    bcs=self.problem.trace_bcs)
        self._assemble_S()
        self.S.force_evaluation()

        split_mixed_op = dict(split_form(M.form))
        split_trace_op = dict(split_form(K.form))
        split_rhs = dict(split_form(F.form))

        A = Tensor(split_mixed_op[(0, 0)])
        B = Tensor(split_mixed_op[(0, 1)])
        C = Tensor(split_mixed_op[(1, 0)])
        D = Tensor(split_mixed_op[(1, 1)])
        K_0 = Tensor(split_trace_op[(0, 0)])
        split_sol = self.broken_solution.split()
        f = Tensor(split_rhs[(1,)])
        sigma = split_sol[0]
        u = split_sol[1]
        lambdar = self.lambdar_sol

        M = D - C * A.inv * B
        u_rec = M.inv * (f + C * A.inv * K_0.T * lambdar)
        self._sub_unknown = create_assembly_callable(u_rec,
                                                     tensor=u)

        sigma_rec = -A.inv * (B * u + K_0.T * lambdar)
        self._elim_unknown = create_assembly_callable(sigma_rec,
                                                      tensor=sigma)

    def _reconstruct(self):
        """
        """
        self._sub_unknown()
        self._elim_unknown()
        udata = self.broken_solution.split()[1].dat
        udata.copy(self.conforming_solution.split()[1].dat)
        reconstruct(self.broken_solution.split()[0],
                    self.conforming_solution.split()[0])

    def solve(self):
        """
        """
        solve(self.S, self.lambdar_sol, self.E,
              solver_parameters=self.params)
        self._reconstruct()
        self._solved = True
        if self.post_process:
            self._post_processing()

    @property
    def computed_solution(self):
        """
        """
        if not self.post_process:
            return self.conforming_solution.split()
        else:
            return self.upp

    def _post_processing(self):
        """
        """
        lambdar = self.lambdar_sol
        u = self.conforming_solution.split()[1]
        d = u.function_space().ufl_element().degree()
        Mk1 = FunctionSpace(self.problem.mesh, "DG", d + 1)
        Tk = self.problem.T
        u_pp = Function(Mk1)
        if d < 2:
            utilde = TrialFunction(Mk1)
            gammar = TestFunction(Tk)
            a = inner(utilde, gammar)*dS + inner(utilde, gammar)*ds
            L = inner(lambdar, gammar)*dS + inner(lambdar, gammar)*ds
            A = Tensor(a)
            F = Tensor(L)
        else:
            Mk_2 = FunctionSpace(self.problem.mesh, "DG", d - 2)
            Wk = Mk_2 * Tk
            utilde = TrialFunction(Mk1)
            v, gammar = TestFunctions(Wk)
            a = inner(utilde, v)*dx + inner(utilde, gammar)*dS
            L = inner(u, v)*dx + inner(lambdar, gammar)*dS
            A = Tensor(a)
            F = Tensor(L)

        assemble(A.inv * F, tensor=u_pp)
        self.upp = u_pp
