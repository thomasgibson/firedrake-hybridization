from firedrake import (SpatialCoordinate, dot, cross, sqrt, atan_2,
                       exp, as_vector, Constant, acos, COMM_WORLD)
import numpy as np


class Gaussian(object):

    def __init__(self,
                 mesh,
                 dir_from_center,
                 radial_dist,
                 sigma_theta,
                 sigma_r,
                 amplitude=1):

        self._mesh = mesh
        self._n0 = dir_from_center
        self._r0 = radial_dist
        self._sigma_theta = sigma_theta
        self._sigma_r = sigma_r
        self._amp = amplitude

        self.x = SpatialCoordinate(mesh)

    @property
    def r(self):
        x = self.x
        return sqrt(x[0]**2 + x[1]**2 + x[2]**2)

    @property
    def theta(self):
        x = self.x
        n0 = self._n0
        return acos(dot(x, n0) / abs(dot(x, n0)))

    @property
    def r_expr(self):
        r = self.r
        r0 = self._r0
        return r - r0

    @property
    def expression(self):
        A = self._amp
        theta = self.theta
        R = self.r_expr
        sigma_theta = self._sigma_theta
        sigma_r = self._sigma_r
        return A*exp(-0.5*((theta/sigma_theta)**2 + (R/sigma_r)**2))


class MultipleGaussians(object):

    def __init__(self, n_gaussians, r_earth, thickness, seed=2097152):

        self._N = n_gaussians
        self._R = r_earth
        self._H = thickness
        self._seed = seed
        self._generate_random_vars()

    def _generate_random_vars(self):
        ns = []
        rs = []
        for i in range(self._N):
            nrm = 0.0

            while (nrm < 0.5) or (nrm > 1.0):
                n = 2*np.random.rand(3) - 1.0
                nrm = np.linalg.norm(n)

            ns.append(as_vector([Constant(k) for k in n]))
            rs.append(Constant(self._R + self._H * np.random.rand()))

        self._random_Ns = ns
        self._random_Rs = rs

    def expression(self, mesh):

        gs = []
        for i, (n, r0) in enumerate(zip(self._random_Ns, self._random_Rs)):
            sigma_theta = 1.0 - 0.5 * (i / self._N)
            sigma_r = (1.0 - 0.5 * (i / self._N)) * self._H
            amplitude = 1.0
            g = Gaussian(mesh, n, r0, sigma_theta, sigma_r, amplitude)
            gs.append(g.expression)

        return sum(gs)
