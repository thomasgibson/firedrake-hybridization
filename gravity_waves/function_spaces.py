from __future__ import absolute_import, print_function, division

from firedrake import *


__all__ = ["generate_function_spaces"]


def generate_function_spaces(mesh, degree=1):
    """Builds the mimetic finite element spaces for the gravity
    wave problem. These spaces include the following:

    (1) W2: An HDiv-conforming velocity space of Raviart-Thomas
            elements.
    (2) W3: A discontinuous L2 pressure space.
    (3) Wb: A buoyancy space constructed from the tensor product
            of a discontinuous L2 element in the horizontal and a
            continuous H1 element in the vertical. This space is
            analogous to the Charney-Phillips C-grid staggered
            finite difference schemes.

    :arg mesh: An Earth-like extruded mesh.
    :arg degree: A ``int`` denoting the order of the
                 finite element spaces.

    Returns: W2, W3 and Wb.
    """

    if mesh._base_mesh.ufl_cell() == quadrilateral:
        hexes = True
    else:
        hexes = False

    # Horizontal elements
    if hexes:
        U1 = FiniteElement("RTCF", quadrilateral, degree)
        U2 = FiniteElement("DQ", quadrilateral, degree - 1)
    else:
        U1 = FiniteElement("RT", triangle, degree)
        U2 = FiniteElement("DG", triangle, degree - 1)

    # Vertical elements
    V0 = FiniteElement("CG", interval, degree)
    V1 = FiniteElement("DG", interval, degree - 1)

    W2_ele = EnrichedElement(HDiv(TensorProductElement(U1, V1)),
                             HDiv(TensorProductElement(U2, V0)))
    W3_ele = TensorProductElement(U2, V1)
    Wb_ele = TensorProductElement(U2, V0)

    W2 = FunctionSpace(mesh, W2_ele)
    W3 = FunctionSpace(mesh, W3_ele)
    Wb = FunctionSpace(mesh, Wb_ele)

    return W2, W3, Wb
