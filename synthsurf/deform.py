from jitfields.pushpull import pull
from .linalg import matvec, lmdiv
import torch


def flow_deform_mesh(vertices, field, affine_mesh=None, affine_field=None, order=3, bound='dct2', prefilter=False):
    """Apply a relative displacement field to a mesh

    Parameters
    ----------
    vertices : (N, D) tensor
        Mesh vertices
    field : (*shape, D) tensor
        Displacement field (in "voxels")
    affine_mesh : (D+1, D+1)
        Matix that maps vertices coordinates to world coordinates
    affine_field : (D+1, D+1)
        Matrix that maps field voxels to world coordinates
    order : 1..7
        Spline order of the flow field
    bound : str
        Boundary conditions of the flow field
    prefilter : bool
        Whether to spline-prefilter the flow field 
        (only use if `field` does not already contain spline coefficients)

    Returns
    -------
    vertices : (N, D) tensor
        Displaced vertices

    """
    ndim = vertices.shape[-1]
    backend = dict(dtype=vertices.dtype, device=vertices.device)

    def pullfn(x, f):
        for _ in range(ndim-1):
            x = x.unsqueeze(0)
        x = pull(x, f, order=order, bound=bound, prefilter=prefilter)
        for _ in range(ndim-1):
            x = x.squeeze(0)
        return x

    if affine_mesh is None:
        affine_mesh = torch.eye(ndim+1, **backend)
    if affine_field is None:
        affine_field = torch.eye(ndim+1, **backend)
    mesh_to_flow = lmdiv(affine_field, affine_mesh)
    flow_to_mesh = lmdiv(affine_mesh, affine_field)

    vertices = matvec(mesh_to_flow[:-1, :-1], vertices).add_(mesh_to_flow[:-1, -1])
    vertices = pullfn(field, vertices).add_(vertices)
    vertices = matvec(flow_to_mesh[:-1, :-1], vertices).add_(flow_to_mesh[:-1, -1])

    return vertices

    
def coord_deform_mesh(vertices, field, affine_mesh=None, affine_field=None, order=3, bound='dct2', prefilter=False):
    """Apply a coordinate field to a mesh

    Parameters
    ----------
    vertices : (N, D) tensor
        Mesh vertices
    field : (*shape, D) tensor
        Coordinate field (in "world coordinates")
    affine_mesh : (D+1, D+1)
        Matix that maps vertices coordinates to world coordinates
    affine_field : (D+1, D+1)
        Matrix that maps flow voxels to world coordinates
    order : 1..7
        Spline order of the coord field
    bound : str
        Boundary conditions of the coord field
    prefilter : bool
        Whether to spline-prefilter the coord field 
        (only use if `field` does not already contain spline coefficients)

    Returns
    -------
    vertices : (N, D) tensor
        Displaced vertices

    """
    ndim = vertices.shape[-1]
    backend = dict(dtype=vertices.dtype, device=vertices.device)

    def pullfn(x, f):
        for _ in range(ndim-1):
            x = x.unsqueeze(0)
        x = pull(x, f, order=order, bound=bound, prefilter=prefilter)
        for _ in range(ndim-1):
            x = x.squeeze(0)
        return x

    if affine_mesh is None:
        affine_mesh = torch.eye(ndim+1, **backend)
    if affine_field is None:
        affine_field = torch.eye(ndim+1, **backend)
    mesh_to_field = lmdiv(affine_field, affine_mesh)

    vertices = matvec(mesh_to_field[:-1, :-1], vertices).add_(mesh_to_field[:-1, -1])
    vertices = pullfn(field, vertices)

    return vertices