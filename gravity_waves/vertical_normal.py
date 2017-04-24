from __future__ import absolute_import, print_function, division

from firedrake import *

import sys
import petsc4py


petsc4py.init(sys.argv)


__all__ = ["VerticalNormal"]


# Credit to Eike Mueller for the kernel code
class VerticalNormal(object):
    """Class for constructing a vertical normal field on a given
    extruded mesh. This is designed for 3D extruded meshes, with
    a base mesh dimension of 2.
    """

    def __init__(self, mesh):
        """Constructor for the VerticalNormal class.

        :arg mesh: An extruded mesh.
        """

        self._mesh = mesh
        self._build_khat()

    @property
    def khat(self):
        """Return the vertical normal."""
        return self._khat

    def _build_khat(self):
        """Generate a kernel that computes the unit normal in
        each cell of the extruded mesh.
        """

        if self._mesh._base_mesh.ufl_cell() == quadrilateral:
            coordinate_fs = FunctionSpace(self._mesh, "DQ", 0)
        else:
            coordinate_fs = FunctionSpace(self._mesh, "DG", 0)

        self._khat = Function(coordinate_fs)

        kernel_code = """build_vertical_normal(double **base_coords,
                                               double **normals) {
            const int ndim=3;
            const int nvert=3;
            double dx[2][ndim];
            double xavg[ndim];
            double n[ndim];

            // Calculate the vector defined by two points
            for (int i=0; i<ndim; ++i) { // Loop over dimensions
                for (int j=0;j<2; ++j) {
                    dx[j][i] = base_coords[j+1][i] - base_coords[0][i];
                }
            }
            // Construct vertical normal
            for (int i=0; i<ndim; ++i) {
                n[i] = dx[0][(i+1)%3]*dx[1][(i+2)%3]
                     - dx[0][(i+2)%3]*dx[1][(i+1)%3];
            }
            // Compute the vector at the center of an edge
            for (int i=0; i<ndim; ++i) { // Loop over dimensions
                xavg[i] = 0.0;
                for (int j=0; j<nvert; ++j) { // Loop over vertices
                    xavg[i] += base_coords[j][i];
                }
            }
            // Calculate vector norm
            double nrm = 0.0;
            double n_dot_xavg = 0.0;
            for (int i=0; i<ndim; ++i) {
                nrm += n[i]*n[i];
                n_dot_xavg += n[i]*xavg[i];
            }
            nrm = sqrt(nrm);
            // Ensure we have the correct orientation
            nrm *= (n_dot_xavg<0?-1:+1);
            // And finally we normalize
            for (int i=0; i<ndim; ++i) {
                normals[0][i] = n[i]/nrm;
            }
        }"""

        kernel = op2.Kernel(kernel_code, "build_vertical_normal")
        base_coords = self._mesh._base_mesh.coordinates
        op2.par_loop(kernel, self._khat.cell_set,
                     base_coords.dat(op2.READ, base_coords.cell_node_map()),
                     self._khat.dat(op2.WRITE, self._khat.cell_node_map()))
