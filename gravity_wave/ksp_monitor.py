import time
import pandas as pd


__all__ = ["KSPMonitor", "KSPMonitorDummy"]


class KSPMonitor(object):
    """Class design from Eike Mueller.

    KSP monitor for writing output.
    """

    def __init__(self, label=None, verbose=2):
        """Constructor for the KSPMonitor

        :arg label: a string name for the solver
        :arg verbose: the verbosity of the monitor.
                      (0): print nothing;
                      (1): only print a summary of the results; and
                      (2): print everything in detail.
        """

        self.label = label or ""
        self.verbose = verbose
        self.initial_residual = 1.0
        self.iterations = []
        self.resnorm = []
        self.resreductions = []
        self.t_start = 0.0
        self.t_start_iter = 0.0
        self.t_finish = 0.0
        self.its = 0

    def __call__(self, ksp, its, rnorm):
        """This method is called by the KSP class and should
        write the output.

        :arg ksp: The calling ksp instance
        :arg its: The current iterations
        :arg rnorm: The current residual norm
        """

        if self.its == 0:
            self.rnorm0 = rnorm
        if self.verbose >= 2:
            reduction = rnorm/self.rnorm0
            s = '  KSP ' + '%20s' % self.label
            s += '  %6d' % its + ' : '
            s += '  %10.6e' % rnorm
            s += '  %10.6e' % reduction
            if self.its > 0:
                s += '  %8.4f' % (rnorm/self.rnorm_old)
            else:
                s += '      ----'
            print(s)
            self.resreductions.append(reduction)
        else:
            self.resreductions.append('Not computed')
        self.its += 1
        if self.its == 1:
            self.t_start_iter = time.clock()
        self.rnorm = rnorm
        self.iterations.append(its)
        self.resnorm.append(rnorm)
        self.rnorm_old = rnorm

    def __enter__(self):
        """Print information at beginning of iteration."""

        self.iterations = []
        self.resnorm = []
        self.its = 0
        if self.verbose < 2:
            print('')
        else:
            s = '  KSP ' + '%20s' % self.label
            s += '    iter             rnrm   rnrm/rnrm_0       rho'
            print(s)
        self.t_start = time.clock()
        self.rnorm = 1.0
        self.rnorm0 = 1.0
        self.rnorm_old = 1.0
        return self

    def __exit__(self, *exc):
        """Print information at end of iteration."""

        self.t_finish = time.clock()
        niter = self.its - 1
        if niter == 0:
            niter = 1
        if self.verbose == 1:
            s = '  KSP ' + '%20s' % self.label
            s += '    iter             rnrm   rnrm/rnrm_0   rho_avg'
            print(s)
            s = '  KSP ' + '%20s' % self.label
            s += '  iter = %6d' % niter + ' : '
            s += '  rnorm = %10.6e' % self.rnorm
            s += '  reduction = %10.6e' % (self.rnorm/self.rnorm0)
            s += '  reduction_rate = %8.4f' % ((self.rnorm/self.rnorm0)**(1./float(niter)))
            print(s)
        if self.verbose >= 1:
            t_elapsed = self.t_finish - self.t_start
            t_elapsed_iter = self.t_finish - self.t_start_iter
            s = '  KSP ' + '%20s' % self.label
            s += ' t_solve = %8.4f s' % t_elapsed
            s += ' t_iter = %8.4f s' % (t_elapsed_iter/niter)
            s += ' [%8.4f s' % (self.t_start_iter - self.t_start) + ']'
            print(s)
            print('')

    def write_to_csv(self):
        """Write collected data to a csv for processing."""

        data = {'niter': self.iterations,
                'residual_norms': self.resnorm,
                'residual_reductions': self.resreductions}
        df = pd.DataFrame(data)
        df.to_csv("ksp_monitor.csv", index=False, mode="w", header=True)


class KSPMonitorDummy(object):

    def __init__(self):
        pass

    def __call__(self, ksp, its, rnorm):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *exc):
        pass

    def write_to_csv(self):
        pass
