from __future__ import absolute_import, print_function, division

from firedrake import *

from function_spaces import generate_function_spaces
from vertical_normal import VerticalNormal


class MixedOperator(object):
    """A class describing the operator of the velocity-pressure sub-system
    of the mixed (Velocity-Pressure-Buoyancy) gravity wave system.

    This is obtained after eliminating the buoyancy unknown to arrive
    at a Helmholtz-like saddle point system.
    """

    def __init__(self, mesh, dt, c, N):
        """Constructor for the MixedSubSystem class.

        :arg mesh: An Earth-like extruded mesh.
        :arg dt: A positive real number denoting the time step size.
        :arg c: A positive real number denoting the speed of sound.
        :arg N: A positive real number denoting the buoyancy frequency.
        """

        super(MixedSubSystem, self).__init__()

        W2, W3, Wb = generate_function_spaces(mesh, degree=1)
        self._mesh = mesh
        self._hdiv_space = W2
        self._L2_space = W3
        self._Wb = Wb
        self._dt = dt
        self._c = c
        self._N = N

        # Constants from eliminating buoyancy
        self._omega_N2 = Constant((0.5*dt*N) ** 2)
        self._dt_half = Constant(0.5*dt)
        self._dt_half_N2 = Constant(0.5*dt*N**2)
        self._dt_half_c2 = Constant(0.5*dt*c**2)

        self._W2W3 = W2 * W3
        u, p = TrialFunctions(self._W2W3)
        v, q = TestFunctions(self._W2W3)

        self._khat = VerticalNormal(self._mesh)

        # Modified velocity mass matrix
        Mutilde = (dot(v, u) + self._omega_N2 * dot(v, self._khat.khat) \
                   dot(self._khat.khat, u)) * dx

        # Off-diagonal blocks
        Qt = (-self._dt_half * div(v) * p) * dx
        Q = self._dt_half_c2 * q * div(u) * dx

        # Pressure mass matrix
        Mp = p * q * dx
        self._bilinear_form = Mutilde + Qt + Q + Mp

        # Boundary conditions
        self._bcs = [DirichletBC(self._hdiv_space, 0.0, "bottom"),
                     DirichletBC(self._hdiv_space, 0.0, "top")]
