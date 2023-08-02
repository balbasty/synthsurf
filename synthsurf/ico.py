import math
import torch
from .surf import vertex_sample, adjacency


def make_icosahedron(device=None, dtype=None, face_dtype=None):
    """Generate an icosahedron whose vertices lie on the unit sphere

    Parameters
    ----------
    device : torch.device, default='cpu'
    dtype : torch.dtype, default=`torch.float32`
    face_dtype : torch.dtype, default=`torch.int64`

    Returns
    -------
    vertices : (12, 3) tensor[dtype]
    faces : (19,) tensor[face_dtype]

    """
    dtype = dtype or torch.get_default_dtype()
    face_dtype = face_dtype or torch.int64
    t = (1 + math.sqrt(5)) / 2
    vertices = [
        [-1,  t,  0],
        [ 1,  t,  0],
        [-1, -t,  0],
        [ 1, -t,  0],
        [ 0, -1,  t],
        [ 0,  1,  t],
        [ 0, -1, -t],
        [ 0,  1, -t],
        [ t,  0, -1],
        [ t,  0,  1],
        [-t,  0, -1],
        [-t,  0,  1],
    ]
    faces = [
        # 5 faces around point 0
        [ 0, 11,  5],
        [ 0,  5,  1],
        [ 0,  1,  7],
        [ 0,  7, 10],
        [ 0, 10, 11],
        # 5 adjacent faces
        [ 1,  5,  9],
        [ 5, 11,  4],
        [11, 10,  2],
        [10,  7,  6],
        [ 7,  1,  8],
        # 5 faces around point 3
        [ 3,  9,  4],
        [ 3,  4,  2],
        [ 3,  2,  6],
        [ 3,  6,  8],
        [ 3,  8,  9],
        # 5 adjacent faces
        [ 4,  9,  5],
        [ 2,  4, 11],
        [ 6,  2, 10],
        [ 8,  6,  7],
        [ 9,  8,  1],
    ]
    vertices = torch.as_tensor(vertices, dtype=dtype, device=device)
    vertices /= math.sqrt(1 + t*t)  # make vertices lie on the unit sphere
    faces = torch.as_tensor(faces, dtype=face_dtype, device=device)
    return vertices, faces


def refine_mesh(vertices, faces, nb_levels=1):
    """Refine a mesh by dividing each triangle into 4 new triangles
    
    Parameters
    ----------
    vertices : (N, 3 + C) tensor[floating]
    faces : (M, 3) tensor[integer]
    nb_levels : int

    Returns
    -------
    vertices : (N', 3 + C) tensor[floating]
    faces : (4*M, 3) tensor[integer]

    """
    # TODO: implement in jitfields

    def refine1(vertices, faces):
        # compute new sizes
        nv = len(vertices)
        nf = len(faces)
        nv_new = nv + (nf * 3) // 2
        nf_new = 4 * nf

        # compute adjacency matrix
        adj = adjacency(len(vertices), faces, dtype=torch.int64)
        adj = adj.sparse_resize_([nv_new, nv_new], 2, 0)

        # allocate larger tensors
        newvertices = vertices.new_empty([nv_new, vertices.shape[-1]])
        newvertices[:nv] = vertices
        newfaces = faces.new_empty([nf_new, faces.shape[-1]])
        
        # loop across faces and subdivide
        nv0 = nv
        for n, face in enumerate(faces):
            verts = [[0, 1], [0, 2], [1, 2]]
            index = [0] * 3
            for k, (i, j) in enumerate(verts):
                if adj[face[i], face[j]] == 2:
                    # insert midpoint as new vertex
                    adj.add_(torch.sparse_coo_tensor(face[[i, j], None], nv0-2, [nv_new, nv_new]))
                    adj.add_(torch.sparse_coo_tensor(face[[j, i], None], nv0-2, [nv_new, nv_new]))
                    newvertices[nv0] = (vertices[face[i]] + vertices[face[j]]).div_(2)
                    index[k] = nv0
                    nv0 += 1
                else:
                    # midpoint already inserted
                    index[k] = adj[face[i], face[j]]
            
            # insert face subdivision
            newface = newfaces[4*n:4*(n+1)]
            newface[0, 0], newface[0, 1], newface[0, 2] = face[0], index[0], index[1]
            newface[1, 0], newface[1, 1], newface[1, 2] = index[0], face[1], index[2]
            newface[2, 0], newface[2, 1], newface[2, 2] = index[1], index[2], face[2]
            newface[3, 0], newface[3, 1], newface[3, 2] = index[0], index[2], index[1]

        newvertices = newvertices[:nv0]
        # assert nv0 == len(newvertices)
        return newvertices, newfaces

    # refine n times
    for _ in range(nb_levels):
        vertices, faces = refine1(vertices, faces)

    return vertices, faces


def project_sphere_(vertices):
    """Project vertices to the unit sphere (inplace)"""
    vertices /= vertices.square().sum(-1, keepdim=True).sqrt_()
    return vertices


def project_sphere(vertices):
    """Project vertices to the unit sphere"""
    vertices = vertices / vertices.square().sum(-1, keepdim=True).sqrt_()
    return vertices


def refine_icosphere(vertices, faces, nb_levels=1):
    """Perform additional subdivisions of the input icosphere
    
    Parameters
    ----------
    vertices : (N, 3 + C) tensor[floating]
    faces : (M,) tensor[integer]
    nb_levels : int

    Returns
    -------
    vertices : (N', 3 + C) tensor[floating]
    faces : (4*M,) tensor[integer]

    """
    vertices, faces = refine_mesh(vertices, faces, nb_levels)
    vertices = project_sphere_(vertices)
    return vertices, faces


def make_icosphere(nb_levels=4, device=None, dtype=None, face_dtype=None):
    """Generate an icosphere by recursively subdivising an icosahedron

    Parameters
    ----------
    nb_levels : int
        Number of subdivisions. 
        The total number of faces will be `20 * (4 ** nb_levels)`.
    device : torch.device, default='cpu'
    dtype : torch.dtype, default=`torch.float32`
    face_dtype : torch.dtype, default=`torch.int64`
    
    Returns
    -------
    vertices : (N, 3) tensor[floating]
    faces : (M,) tensor[integer]

    """
    vertices, faces = make_icosahedron(device, dtype, face_dtype)
    vertices, faces = refine_icosphere(vertices, faces, nb_levels)
    return vertices, faces