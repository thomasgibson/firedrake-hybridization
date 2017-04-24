from __future__ import absolute_import, print_function, division

from firedrake import UnitSquareMesh, UnitCubeMesh, ExtrudedMesh


__all__ = ["generate_2d_square_mesh",
           "generate_3d_cube_mesh",
           "generate_3d_cube_extr_mesh"]


def generate_2d_square_mesh(r, quadrilateral=False):
    """Generates a firedrake mesh of a unit square domain:
    [0, 1] x [0, 1].

    :arg r: An ``int`` for computing the mesh resolution.
    :arg quadrilateral: A ``bool`` denoting whether to use
                        quadrilateral mesh cells.

    Returns: A Firedrake mesh.
    """
    return UnitSquareMesh(2 ** r, 2 ** r, quadrilateral=quadrilateral)


def generate_3d_cube_mesh(r, quadrilateral=False):
    """Generates a firedrake mesh of a unit cube domain:
    [0, 1] x [0, 1] x [0, 1].

    :arg r: An ``int`` for computing the mesh resolution.
    :arg quadrilateral: A ``bool`` denoting whether to use
                        quadrilateral-prism mesh cells.

    Returns: A Firedrake mesh.
    """
    return UnitCubeMesh(2 ** r, 2 ** r, 2 ** r, quadrilateral=quadrilateral)


def generate_3d_cube_extr_mesh(r, quadrilateral=False):
    """Generates a firedrake mesh of a unit cube domain:
    [0, 1] x [0, 1] x [0, 1] using an extruded cell set.

    The layer height and number is computed based on the
    mesh resolution.

    :arg r: An ``int`` for computing the mesh resolution.
    :arg quadrilateral: A ``bool`` denoting whether to use
                        quadrilateral-prism mesh cells.

    Returns: A Firedrake extruded mesh.
    """
    base = generate_2d_square_mesh(r, quadrilateral=quadrilateral)
    layers = 2 ** r
    layer_height = 1.0 / layers
    return ExtrudedMesh(base, layers=layers, layer_height=layer_height)


# TODO: Add sphere mesh for the gravity wave test case
