from firedrake import *


__all__ = ["P1HMGSolver"]


class P1HMGSolver(object):
    """
    """

    def __init__(self, S, fine_solution, E,
                 mesh_hierarchy=None, mg_type="amg"):
        """
        """
        if mg_type == "gmg":
            assert mesh_hierarchy is not None

        self.lambda_f = fine_solution
        self.trace_operator = S
        self.trace_rhs = E
        self._WT = self.lambda_f.function_space()
        self._CG1 = FunctionSpace(self._WT.mesh(), "CG", 1)
        self._p1_sol = Function(self._CG1)
        self._p1_f = Function(self._CG1)
        self._p1_r = Function(self._CG1)
        self._trace_r = Function(self._WT)
        self._update = Function(self._WT)

    def presmooth_trace(self):
        E_f = self.trace_rhs
        S_f = self.trace_operator

        # Pre-smooth with a 3-point Jacobi method
        solve(S_f, self.lambda_f, E_f,
              solver_parameters={'ksp_type': 'richardson',
                                 'ksp_max_it': 3,
                                 'ksp_convergence_test': 'skip',
                                 'ksp_initial_guess_nonzero': True,
                                 'pc_type': 'jacobi'})

    def postsmooth_trace(self):
        E_f = self.trace_rhs
        S_f = self.trace_operator

        # Post-smooth with a 5-point Jacobi method
        solve(S_f, self.lambda_f, E_f,
              solver_parameters={'ksp_type': 'richardson',
                                 'ksp_max_it': 5,
                                 'ksp_convergence_test': 'skip',
                                 'ksp_initial_guess_nonzero': True,
                                 'pc_type': 'jacobi'})

    def restrict_to_p1(self, r):
        project(r, self._p1_r,
                solver_parameters={'ksp_type': 'cg',
                                   'ksp_rtol': 1.0E-10,
                                   'pc_type': 'bjacobi',
                                   'sub_pc_type': 'ilu'})

    def prolong_to_trace(self, u):
        gammar = TestFunction(self._WT)
        lambdar = TrialFunction(self._WT)
        a = (gammar('+')*lambdar('+')*(dS_v + dS_h) +
             gammar*lambdar*ds_tb)
        L = gammar('+')*u('+')*(dS_v + dS_h) + gammar*u*ds_tb
        solve(a, self._update, L,
              solver_parameters={'ksp_type': 'cg',
                                 'ksp_rtol': 1.0E-10,
                                 'pc_type': 'bjacobi',
                                 'sub_pc_type': 'ilu'})

    def solve(self):

        # Pre-smooth trace solution
        self.presmooth_trace()

        S_f = self.trace_operator
        E_f = self.trace_rhs

        # Doesn't work (firedrake matrix * function not right)
        r_f = assemble(E_f - S_f * self.lambda_f)
        self.restrict_to_p1(r_f)

        # Restricted residual
        r_p1 = self._p1_r

        # MG step for the P1 problem
        u = TrialFunction(self._CG1)
        v = TestFunction(self._CG1)
        a = inner(grad(v), grad(u))*dx
        L = inner(v, r_p1)*dx
        solve(a, self._p1_f, L,
              solver_parameters={{'ksp_type': 'preonly',
                                  'pc_type': 'hypre',
                                  'pc_hypre_type': 'boomeramg',
                                  'pc_hypre_boomeramg_no_CF': False,
                                  'pc_hypre_boomeramg_coarsen_type': 'HMIS',
                                  'pc_hypre_boomeramg_interp_type': 'ext+i',
                                  'pc_hypre_boomeramg_P_max': 0,
                                  'pc_hypre_boomeramg_agg_nl': 0,
                                  'pc_hypre_boomeramg_max_level': 2,
                                  'pc_hypre_boomeramg_strong_threshold': 0.25}})

        # Prolong to trace
        self.prolong_to_trace(self._p1_f)

        # Update
        self.lambda_f += self._update

        # Post-smooth
        self.post_smooth()
