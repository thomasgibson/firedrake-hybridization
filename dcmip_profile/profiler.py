from gusto.configuration import logger
from gusto.timeloop import CrankNicolson as GCN
from gusto.linear_solvers import HybridizedCompressibleSolver
from pyop2.profiling import timed_stage
from mpi4py import MPI
from firedrake.petsc import PETSc
from firedrake import COMM_WORLD

import os
import pandas as pd


__all__ = ["Profiler"]


class Profiler(GCN):
    """
    Profiles only the implicit linear solver in a Gusto model.

    WARNING: Do not use when running a full model. This is only
             for profiling purposes and therefore doesn't include
             all Gusto stages needed for a complete simulation.
    """

    def __init__(self, state, advected_fields,
                 linear_solver, forcing,
                 diffused_fields=None, physics_list=None,
                 prescribed_fields=None,
                 label=None):

        super(Profiler, self).__init__(state=state,
                                       advected_fields=advected_fields,
                                       linear_solver=linear_solver,
                                       forcing=forcing,
                                       diffused_fields=diffused_fields,
                                       physics_list=physics_list,
                                       prescribed_fields=prescribed_fields)

        if isinstance(self.linear_solver, HybridizedCompressibleSolver):
            self.hybridization = True
        else:
            self.hybridization = False

        if label:
            tag = "profile_%s" % label
        else:
            tag = "profile"

        self.tag = tag
        self._warm_run = False

    def implicit_step(self):

        state = self.state

        # xrhs is the residual which goes in the linear solve
        state.xrhs.assign(0.0)

        state.xrhs -= state.xnp1

        if not self._warm_run:
            logger.info("Cold run, warming up solver.")
            self.linear_solver.solve()
            state.dy.assign(0.0)
            self._warm_run = True
            logger.info("Solver warmed up.")

        logger.info("Profiling linear solver.")

        if self.hybridization:
            solver = self.linear_solver.hybridized_solver
        else:
            solver = self.lienar_solver.urho_solver

        solver.snes.setConvergenceHistory()
        solver.snes.ksp.setConvergenceHistory()

        with timed_stage("Implicit solve"):
            self.linear_solver.solve()

        state.xnp1 += state.dy

        self.extract_ksp_info(solver)

        self._apply_bcs()

    def run(self, t, tmax, pickup=False):

        logger.info("Profiling one time-step.")
        state = self.state
        state.xnp1.assign(state.xn)
        self.implicit_step()
        state.xb.assign(state.xn)
        state.xn.assign(state.xnp1)
        logger.info("Profile complete for  one time-step.")

    def extract_ksp_info(self, solver):

        problem = solver._problem
        x = problem.u

        PETSc.Log.Stage("Implicit solve").push()

        snes = PETSc.Log.Event("SNESSolve").getPerfInfo()
        ksp = PETSc.Log.Event("KSPSolve").getPerfInfo()
        pcsetup = PETSc.Log.Event("PCSetUp").getPerfInfo()
        pcapply = PETSc.Log.Event("PCApply").getPerfInfo()
        jac_eval = PETSc.Log.Event("SNESJacobianEval").getPerfInfo()
        residual = PETSc.Log.Event("SNESFunctionEval").getPerfInfo()

        comm = problem.comm
        snes_time = comm.allreduce(snes["time"], op=MPI.SUM) / comm.size
        ksp_time = comm.allreduce(ksp["time"], op=MPI.SUM) / comm.size
        pcsetup_time = comm.allreduce(pcsetup["time"], op=MPI.SUM) / comm.size
        pcapply_time = comm.allreduce(pcapply["time"], op=MPI.SUM) / comm.size
        jac_time = comm.allreduce(jac_eval["time"], op=MPI.SUM) / comm.size
        res_time = comm.allreduce(residual["time"], op=MPI.SUM) / comm.size

        num_cells = comm.allreduce(x.function_space().mesh.cell_set.size,
                                   op=MPI.SUM)
        total_dofs = x.dof_dset.layout_vec.getSize()

        if self.hybridization:
            ksp = solver.snes.ksp.getPC().getPythonContext().condensed_ksp
        else:
            ksp = solver.snes.ksp

        if COMM_WORLD.rank == 0:
            results = "profiling/"
            if not os.path.exists(os.path.dirname(results)):
                os.makedirs(os.path.dirname(results))

            data = {"SNESSolve": snes_time,
                    "KSPSolve": ksp_time,
                    "PCSetUp": pcsetup_time,
                    "PCApply": pcapply_time,
                    "SNESJacobianEval": jac_time,
                    "SNESFunctionEval": res_time,
                    "num_processes": problem.comm.size,
                    "num_cells": num_cells,
                    "total_dofs": total_dofs,
                    "ksp_iters": ksp.getIterationNumber()}

            df = pd.DataFrame(data, index=[0])
            result_file = results + "%s.csv" % self.tag
            df.to_csv(result_file, index=False, mode="w", header=True)

        PETSc.Log.Stage("Implicit solve").pop()
