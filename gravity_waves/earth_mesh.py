from __future__ import absolute_import, print_function, division

from firedrake import *


__all__ = ["generate_earth_mesh"]


def generate_earth_mesh(r_level, num_layers, thickness, hexes=False):
    """Generates an Earth-like spherical mesh for the gravity wave
    problem.

    :arg r_level: An ``int`` denoting the number of refinement levels.
    :arg num_layers: An ``int`` denoting the number of mesh layers.
    :arg thickness: The thickness of the spherical shell (in meters).
    :arg hexes: A ``bool`` indicating whether to generate a hexahedral mesh.

    Returns: A Firedrake extruded spherical mesh.
    """

    earth_radius = 6.371e6
    layer_height = thickness / num_layers

    if hexes:
        spherical_base = CubedSphereMesh(earth_radius,
                                         refinement_level=r_level)
    else:
        spherical_base = IcosahedralSphereMesh(earth_radius,
                                               refinement_level=r_level)

    earth_mesh = ExtrudedMesh(spherical_base, layers=num_layers,
                              layer_height=layer_height,
                              extrusion_type="radial")
    return earth_mesh
