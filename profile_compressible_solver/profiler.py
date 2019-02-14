from gusto.configuration import logger
from gusto.timeloop import CrankNicolson as GCN
from gusto.linear_solvers import HybridizedCompressibleSolver
from mpi4py import MPI
from firedrake.petsc import PETSc
from firedrake import COMM_WORLD
from pyop2.profiling import timed_stage

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

    def __init__(self, parameterinfo, state, advected_fields,
                 linear_solver, forcing,
                 diffused_fields=None, physics_list=None,
                 prescribed_fields=None,
                 suppress_data_output=False):

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

        tag = "profile_dx%s_dz%s" % (int(parameterinfo.deltax),
                                     int(parameterinfo.deltaz))

        self.tag = tag
        self._warm_run = False
        self.parameter_info = parameterinfo
        self.suppress_data_output = suppress_data_output

    def semi_implicit_step(self):

        state = self.state
        dt = state.timestepping.dt
        alpha = state.timestepping.alpha

        # Do some forcing and advection to get some crunchy data
        with timed_stage("Apply forcing terms"):
            self.forcing.apply((1-alpha)*dt, state.xn, state.xn,
                               state.xrhs, implicit=False)

        logger.info("Finished forcing. Profiling linear solver.")

        if self.hybridization:
            solver = self.linear_solver.hybridized_solver
        else:
            solver = self.linear_solver.urho_solver

        solver.snes.setConvergenceHistory()
        solver.snes.ksp.setConvergenceHistory()

        with PETSc.Log.Stage("linear_solve"):
            solver.solve()
            if not self.suppress_data_output:
                self.extract_ksp_info(solver)

        state.xnp1 += state.dy

        self._apply_bcs()

    def run(self, t, tmax, pickup=False):

        logger.info("Profiling one time-step.")
        state = self.state
        state.xnp1.assign(state.xn)
        self.semi_implicit_step()
        state.xb.assign(state.xn)
        state.xn.assign(state.xnp1)
        logger.info("Profile complete for  one time-step.")

    def extract_ksp_info(self, solver):

        problem = solver._problem
        x = problem.u
        comm = x.function_space().mesh().comm

        snes = PETSc.Log.Event("SNESSolve").getPerfInfo()
        ksp = PETSc.Log.Event("KSPSolve").getPerfInfo()
        pcsetup = PETSc.Log.Event("PCSetUp").getPerfInfo()
        pcapply = PETSc.Log.Event("PCApply").getPerfInfo()
        jac_eval = PETSc.Log.Event("SNESJacobianEval").getPerfInfo()
        residual = PETSc.Log.Event("SNESFunctionEval").getPerfInfo()

        snes_time = comm.allreduce(snes["time"], op=MPI.SUM) / comm.size
        ksp_time = comm.allreduce(ksp["time"], op=MPI.SUM) / comm.size
        pcsetup_time = comm.allreduce(pcsetup["time"], op=MPI.SUM) / comm.size
        pcapply_time = comm.allreduce(pcapply["time"], op=MPI.SUM) / comm.size
        jac_time = comm.allreduce(jac_eval["time"], op=MPI.SUM) / comm.size
        res_time = comm.allreduce(residual["time"], op=MPI.SUM) / comm.size

        num_cells = comm.allreduce(x.function_space().mesh().cell_set.size,
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

            data = {
                "SNESSolve": snes_time,
                "KSPSolve": ksp_time,
                "PCSetUp": pcsetup_time,
                "PCApply": pcapply_time,
                "SNESJacobianEval": jac_time,
                "SNESFunctionEval": res_time,
                "num_processes": comm.size,
                "num_cells": num_cells,
                "total_dofs": total_dofs,
                "ksp_iters": ksp.getIterationNumber(),
                "outer_solver_type": self.parameter_info.solver_type,
                "inner_solver_type": self.parameter_info.inner_solver_type,
                "dt": self.parameter_info.dt,
                "deltax": self.parameter_info.deltax,
                "deltaz": self.parameter_info.deltaz,
                "horizontal_courant": self.parameter_info.horizontal_courant,
                "vertical_courant": self.parameter_info.vertical_courant,
                "model_family": self.parameter_info.family,
                "model_degree": self.parameter_info.model_degree,
                "mesh_degree": self.parameter_info.mesh_degree
            }

            df = pd.DataFrame(data, index=[0])
            result_file = results + "%s_%s_%s_data.csv" % (
                self.tag,
                self.parameter_info.solver_type,
                self.parameter_info.inner_solver_type
            )
            df.to_csv(result_file, index=False, mode="w", header=True)
