from firedrake import *


__all__ = ["construct_spaces"]


def construct_spaces(mesh, order=1):
    """Builds the compatible finite element spaces
    for the linear compressible gravity wave system.

    The following spaces are constructed:

    W2: The HDiv velocity space
    W3: The L2 pressure space
    Wb: The Charney-Phillips space for buoyancy. This
        space is constructed from an L2 horizontal
        finite element space augmented with a vertical
        H1-conforming element.

    :arg mesh: An extruded mesh.
    :arg order: Integer denoting the order of the
                element spaces. Default is `1`, which
                corresponds to a lowest-order method.
    """

    assert order >= 1

    # Horizontal elements
    if mesh._base_mesh.ufl_cell().cellname() == 'quadrilateral':
        U1 = FiniteElement('RTCF', quadrilateral, order)
        U2 = FiniteElement('DQ', quadrilateral, order - 1)

    elif mesh._base_mesh.ufl_cell().cellname() == 'triangle':
        U1 = FiniteElement('RT', triangle, order)
        U2 = FiniteElement('DG', triangle, order - 1)

    else:
        assert mesh._base_mesh.ufl_cell().cellname() == 'interval'
        U1 = FiniteElement('CG', interval, order)
        U2 = FiniteElement('DG', interval, order - 1)

    # Vertical elements
    V0 = FiniteElement('CG', interval, order)
    V1 = FiniteElement('DG', interval, order - 1)

    # HDiv element
    W2_ele_h = HDiv(TensorProductElement(U1, V1))
    W2_ele_v = HDiv(TensorProductElement(U2, V0))
    W2_ele = W2_ele_h + W2_ele_v

    # L2 element
    W3_ele = TensorProductElement(U2, V1)

    # Charney-Phillips element
    Wb_ele = TensorProductElement(U2, V0)

    W2 = FunctionSpace(mesh, W2_ele)
    W3 = FunctionSpace(mesh, W3_ele)
    Wb = FunctionSpace(mesh, Wb_ele)
    W2v = FunctionSpace(mesh, W2_ele_v)

    return W2, W3, Wb, W2v
