import torch
import math
from .linalg import dot, outer, lmdiv, matvec

# WARNING: most of these functions assume a closed surface
#          (= isomorphic to a sphere)


def vertex_sample(vertices, indices):
    """Sample values at vertices

    Parameters
    ----------
    vertices : (N, *fshape) tensor
        Mapping from vertex to features
    indices : (*ishape) tensor[long]
        Vertex indices

    Returns
    -------
    sampled_vertices : (*ishape, *fshape)

    """
    N, *fshape = vertices.shape
    ishape = indices.shape
    indices = indices.long()
    for _ in range(len(ishape)):
        vertices = vertices.unsqueeze(1)
    vertices = vertices.expand([N, *ishape, *fshape])
    for _ in range(len(fshape)):
        indices = indices.unsqueeze(-1)
    indices = indices.expand([1, *ishape, *fshape])
    vertices = vertices.gather(0, indices)[0]
    return vertices


def vertex_scatter(indices, values, n):
    """Scatter values on vertices

    Parameters
    ----------
    indices : (*shape) tensor[long]
        Vertex indices
    values : (*shape, *fshape) tensor
        Values to scatter
    n : int
        Number of vertices

    Returns
    -------
    scattered_values : (N, *fshape) tensor

    """
    values = torch.as_tensor(values, device=indices.device)
    values = values.unsqueeze(max(0, indices.dim() - values.dim()))
    val_dim = values.dim()
    ind_dim = indices.dim()
    values = values.expand([*indices.shape, *values.shape[ind_dim:]])
    out = values.new_zeros([n, *values.shape[ind_dim:]])
    out = out.expand([n, *values.shape[ind_dim:]])
    for _ in range(val_dim - ind_dim):
        out = out.unsqueeze(1)
    while indices.ndim < values.ndim:
        indices = indices.unsqueeze(-1)
    out.index_add_(0, indices, values)
    for _ in range(val_dim - ind_dim):
        out = out.squeeze(1)
    return out


def vertex_scatter_add_(out, indices, values):
    """Scatter values on vertices

    Parameters
    ----------
    out : (N, *fshape) tensor
        Tensor to scatter into
    indices : (*shape) tensor[long]
        Vertex indices
    values : (*shape, *fshape) tensor
        Values to scatter

    Returns
    -------
    out : (N, *fshape) tensor

    """
    values = torch.as_tensor(values, device=out.device, dtype=out.dtype)
    fdim = out.ndim - 1
    ishape = indices.shape
    n, *fshape = out.shape
    values = values.expand([*ishape, *fshape]).reshape([-1, *fshape])
    for _ in range(fdim):
        indices = indices.unsqueeze(-1)
    indices = indices.expand([*ishape, *fshape]).reshape([-1, *fshape])
    out.scatter_add_(0, indices, values)
    return out


def face_barycenter(coord, faces):
    """Compute the barycenter of each face

    Parameters
    ----------
    coord : (N, D) tensor
        Vertices coordinates
    faces : (M, K) tensor[long]
        Vertices indices of each face

    Returns
    -------
    bary : (M, D) tensor
        Barycenter of each face

    """
    bary = vertex_sample(coord, faces).transpose(-1, -2)   # MDK
    bary = bary.mean(-1)
    return bary


def face_normal(coord, faces, normalize=True):
    """Compute the normal vector of each face

    Parameters
    ----------
    coord : (N, D) tensor
        Vertices coordinates
    faces : (M, K) tensor[long]
        Vertices indices of each face
    normalize : bool, default=True
        Normalize the vector. Otherwise, its norm equals the face's area.

    Returns
    -------
    norm : (M, D) tensor
        Normal vector

    """
    dim = coord.shape[-1]

    if not normalize:
        faces = faces[:, :dim]
    vert = vertex_sample(coord, faces).transpose(-1, -2) # MDK

    if dim == 2:
        norm = vert[..., 1] - vert[..., 0]
        norm = norm.flip(-1)
        norm[..., 0].neg_()
    elif dim == 3:
        a = vert[..., 1] - vert[..., 0]
        b = vert[..., 2] - vert[..., 0]
        norm = torch.cross(a, b).div_(2)
    else:
        raise NotImplementedError('`norm` is only implemented in '
                                  'dimension 2 and 3.')

    polydim = faces.shape[-1]
    if normalize or polydim > dim:
        norm /= dot(norm, norm, keepdim=True).sqrt_()
    if polydim > dim:
        # https://math.stackexchange.com/questions/3207981
        area = 0
        for k in range(1, polydim-1):
            a = vert[..., k] - vert[..., 0]
            b = vert[..., k+1] - vert[..., 0]
            area += torch.cross(a, b)
        area = dot(area, area, keepdim=True).sqrt_().div_(2)
        norm *= area

    return norm


def face_area(coord, faces):
    """Compute the area of each face

    Parameters
    ----------
    coord : (N, D) tensor
        Vertices coordinates
    faces : (M, K) tensor[long]
        Vertices indices of each face

    Returns
    -------
    area : (M, [D]) tensor
        Area of each face

    """
    # https://math.stackexchange.com/questions/3207981
    polydim = faces.shape[-1]
    vert = vertex_sample(coord, faces).transpose(-1, -2)   # MDK

    area = 0
    for k in range(1, polydim - 1):
        a = vert[..., k] - vert[..., 0]
        b = vert[..., k + 1] - vert[..., 0]
        area += torch.cross(a, b)
    return dot(area, area).sqrt_().div_(2)


def vertex_normal(coord, faces):
    """Compute the normal vector to each vertex

    Notes
    -----
    This is done by averaging the neighboring face's normals weighted
    by their area.

    Parameters
    ----------
    coord : (N, D) tensor
        Vertices coordinates
    faces : (M, K) tensor[long]
        Vertices indices of each face

    Returns
    -------
    norm : (N, D) tensor
        Vertices normals

    """
    faces = faces.long()
    vnormals = torch.zeros_like(coord)
    fnormals = face_normal(coord, faces, normalize=False)

    vnormals.index_add_(0, faces[:, 0], fnormals)
    vnormals.index_add_(0, faces[:, 1], fnormals)
    vnormals.index_add_(0, faces[:, 2], fnormals)
    vnormals /= dot(vnormals, vnormals, keepdim=True).sqrt_()
    return vnormals


def _quad_fit(coord, faces, norm, fixdf=2):
    # Fit (least-square) a 2d quadratic on the tangent plane of each vertex
    # - (x, y) are the positions of neighbouring vertices projected to
    #   the tangent plane
    # - z is the distance from the neighbouring vertices to the tangent plane
    #
    # if fixdf == 0: fit quadratic + linear + constant terms
    # if fixdf == 1: fit quadratic + tangent term (force constant == 0)
    # if fixdf == 2: fit quadratic (force tangent ==0, constant == 0)
    #
    # fixdf == 2 can always be fitted on a closed surface since the minimum
    # valence if 3. When fixdf < 2, some vertices may need higher order
    # neighbors, but this option is not currently implemented.
    #
    # Currently, we only fit using the first ring neighbors.

    N, D = coord.shape
    M, K = faces.shape

    # define a basis of the tangent plane
    u = norm.clone()
    u[:, -1] = 0
    u[:, :2] = u[:, :2].fliplr()
    u[:, 0].neg_()
    v = norm.clone()
    v[:, :2] *= v[:, -1:]
    v[:, -1] = dot(norm[:, :2], norm[:, :2]).neg_()
    b = torch.stack([u, v], 1)
    b /= dot(b, b, keepdim=True).sqrt_()
    # [N, 2, 3]

    # compute sufficient statistics up to order 4

    coord2 = outer(coord, coord)
    coord3 = outer(coord2, coord[:, None])
    coord4 = outer(coord3, coord[:, None, None])

    ss0 = faces.new_zeros([N])
    ss1 = coord.new_zeros([N, 3])
    ss2 = coord.new_zeros([N, 3, 3])
    ss3 = coord.new_zeros([N, 3, 3, 3])
    ss4 = coord.new_zeros([N, 3, 3, 3, 3])

    fvertices = vertex_sample(coord, faces)
    one = ss0.new_ones([1]).expand([M])
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            ss0.index_add_(0, faces[:, i], one)
            tmp = fvertices[:, j]
            ss1.index_add_(0, faces[:, i], tmp)
            tmp = outer(tmp, fvertices[:, j])
            ss2.index_add_(0, faces[:, i], tmp)
            tmp = outer(tmp, fvertices[:, None, j])
            ss3.index_add_(0, faces[:, i], tmp)
            tmp = outer(tmp, fvertices[:, None, None, j])
            ss4.index_add_(0, faces[:, i], tmp)
    # correct overcounting
    ss0.floor_divide_(2)
    ss1.div_(2)
    ss2.div_(2)
    ss3.div_(2)
    ss4.div_(2)

    # TODO: option to add second (and higher) ring neighbors

    # subtract origin

    coord4 *= ss0[:, None, None, None, None]
    ss4 += coord4
    del coord4
    coord3x = outer(coord3, ss1[:, None, None])
    ss4 -= coord3x
    ss4 -= coord3x.transpose(-1, -2)
    ss4 -= coord3x.transpose(-1, -3)
    ss4 -= coord3x.transpose(-1, -4)
    del coord3x
    coord2x2 = outer(coord2[:, :, None], ss2[:, None])
    ss4 += coord2x2
    ss4 += coord2x2.transpose(-1, -3)
    ss4 += coord2x2.transpose(-1, -4)
    ss4 += coord2x2.transpose(-2, -3)
    ss4 += coord2x2.transpose(-2, -4)
    ss4 += coord2x2.transpose(-2, -4).transpose(-1, -3)
    del coord2x2
    coordx3 = outer(ss3, coord[:, None, None])
    ss4 -= coordx3
    ss4 -= coordx3.transpose(-1, -2)
    ss4 -= coordx3.transpose(-1, -3)
    ss4 -= coordx3.transpose(-1, -4)
    del coordx3

    coord3 *= ss0[:, None, None, None]
    ss3 -= coord3
    del coord3
    coord2x = outer(coord2, ss1[:, None])
    ss3 += coord2x
    ss3 += coord2x.transpose(-1, -2)
    ss3 += coord2x.transpose(-1, -3)
    del coord2x
    coordx2 = outer(ss2, coord[:, None])
    ss3 -= coordx2
    ss3 -= coordx2.transpose(-1, -2)
    ss3 -= coordx2.transpose(-1, -3)
    del coordx2

    coord2 *= ss0[:, None, None]
    ss2 += coord2
    del coord2
    coordx = outer(coord, ss1)
    ss2 -= coordx
    ss2 -= coordx.transpose(-1, -2)
    del coordx

    ss1 -= coord * ss0[:, None]

    # project on tangent plane

    P2 = lambda x: x.transpose(-1, -2)
    P3 = lambda x: x.transpose(-1, -2).transpose(-2, -3)
    P4 = lambda x: x.transpose(-1, -2).transpose(-2, -3).transpose(-3, -4)

    ss0z = dot(norm, ss1)
    ss1 = matvec(b, ss1)
    ss2 = P2(matvec(b[:, None], ss2))
    ss1z = dot(norm[:, None], ss2)               # distance to plane
    ss2 = P2(matvec(b[:, None], ss2))
    ss3 = P3(matvec(b[:, None, None], ss3))
    ss3 = P3(matvec(b[:, None, None], ss3))
    ss2z = dot(norm[:, None, None], ss3)         # distance to plane
    ss3 = P3(matvec(b[:, None, None], ss3))
    ss4 = P4(matvec(b[:, None, None, None], ss4))
    ss4 = P4(matvec(b[:, None, None, None], ss4))
    ss4 = P4(matvec(b[:, None, None, None], ss4))
    ss4 = P4(matvec(b[:, None, None, None], ss4))

    # compute matrices for pseudo-inverse
    df = 3 if fixdf == 2 else 5 if fixdf == 1 else 6
    xx = ss1.new_empty([N, df, df])
    xx[:, 0, 0] = ss4[:, 0, 0, 0, 0]
    xx[:, 1, 1] = ss4[:, 1, 1, 1, 1]
    xx[:, 0, 1] = xx[:, 1, 0] = xx[:, 2, 2] = ss4[:, 0, 0, 1, 1]
    xx[:, 0, 2] = xx[:, 2, 0] = ss4[:, 0, 0, 0, 1]
    xx[:, 1, 2] = xx[:, 2, 1] = ss4[:, 0, 1, 1, 1]
    if df > 3:
        xx[:, 3, 3] = ss2[..., 0, 0]
        xx[:, 4, 4] = ss2[..., 1, 1]
        xx[:, 3, 4] = xx[:, 4, 3] = ss2[..., 0, 1]
        xx[:, 0, 3] = xx[:, 3, 0] = ss3[..., 0, 0, 0]
        xx[:, 0, 4] = xx[:, 4, 0] = xx[:, 2, 3] = xx[:, 3, 2] = ss3[..., 0, 0, 1]
        xx[:, 1, 3] = xx[:, 3, 1] = xx[:, 2, 4] = xx[:, 4, 2] = ss3[..., 0, 1, 1]
        xx[:, 1, 4] = xx[:, 4, 1] = ss3[..., 1, 1, 1]
        if df > 5:
            xx[:, 5, 5] = 1
            xx[:, 0, 5] = xx[:, 5, 0] = ss2[:, 0, 0]
            xx[:, 1, 5] = xx[:, 5, 1] = ss2[:, 1, 1]
            xx[:, 2, 5] = xx[:, 5, 2] = ss2[:, 0, 1]
            xx[:, 3, 5] = xx[:, 5, 3] = ss1[:, 0]
            xx[:, 4, 5] = xx[:, 5, 4] = ss1[:, 1]
    xx[:, :, 2] *= 2
    xx[:, 2, :] *= 2

    # It's a bit easier to visualize the matrix form:
    # xx = [[ss4[:, 0, 0, 0, 0], ss4[:, 0, 0, 1, 1], ss4[:, 0, 0, 0, 1], ss3[:, 0, 0, 0], ss3[:, 0, 0, 1], ss2[:, 0, 0]],
    #       [ss4[:, 0, 0, 1, 1], ss4[:, 1, 1, 1, 1], ss4[:, 0, 1, 1, 1], ss3[:, 0, 1, 1], ss3[:, 1, 1, 1], ss2[:, 1, 1]],
    #       [ss4[:, 0, 0, 0, 1], ss4[:, 0, 1, 1, 1], ss4[:, 0, 0, 1, 1], ss3[:, 0, 0, 1], ss3[:, 0, 1, 1], ss2[:, 0, 1]],
    #       [ss3[:, 0, 0, 0],    ss3[:, 0, 1, 1],    ss3[:, 0, 0, 1],    ss2[:, 0, 0],    ss2[:, 0, 1],    ss1[:, 0]],
    #       [ss3[:, 0, 0, 1],    ss3[:, 1, 1, 1],    ss3[:, 0, 1, 1],    ss2[:, 0, 1],    ss2[:, 1, 1],    ss1[:, 1]],
    #       [ss2[:, 0, 0],       ss2[:, 1, 1],       ss2[:, 0, 1],       ss1[:, 0],       ss1[:, 1],       1],]
    del ss4, ss3, ss2, ss1

    xy = ss1z.new_empty([N, df])
    xy[:, 0] = ss2z[:, 0, 0]
    xy[:, 1] = ss2z[:, 1, 1]
    xy[:, 2] = ss2z[:, 0, 1]
    xy[:, 2] *= 2
    if df > 3:
        xy[:, 3] = ss1z[:, 0]
        xy[:, 4] = ss1z[:, 1]
        if df > 5:
            xy[:, 5] = ss0z

    # It's a bit easier to visualize the vector form:
    # xy = [ss2z[:, 0, 0], ss2z[:, 1, 1], ss2z[:, 0, 1], ss1z[:, 0], ss1z[:, 1], ss0z]
    del ss2z, ss1z, ss0z

    coeff = lmdiv(xx, xy.unsqueeze(-1)).squeeze(-1)

    return coeff, b


def _vertex_curv(coord, faces, norm=None, fixdf=0, dir=True):
    # compute the SVD of the quadratic fit + project vectors back to 3D

    if norm is None:
        norm = vertex_normal(coord, faces)

    # perform quadratic fit in each vertex
    coeff, basis = _quad_fit(coord, faces, norm, fixdf=fixdf)

    g = c = None

    # decompose and reparameterize as: 0.5 * (x - g)'H(x - g) + c
    huu, hvv, huv, *coeff = coeff.unbind(-1)
    h = huu.new_empty([len(huu), 2, 2])
    h[:, 0, 0] = huu
    h[:, 1, 1] = hvv
    h[:, 0, 1] = h[:, 1, 0] = huv
    h.mul_(2)
    s, u = torch.symeig(h, eigenvectors=dir)
    # NOTE: mean curvature could be obtained from the trace alone.

    if coeff:
        gu, gv, *coeff = coeff
        g = torch.stack([gu, gv], -1)
        g = lmdiv(h, g.unsqueeze(-1)).squeeze(-1).neg_()

        if coeff:
            c = coeff[0]
            c -= dot(g, matvec(h, g)).div_(2)

    # reparameterize in 3D
    if dir:
        u = matvec(u[:, :, None, :], basis[:, None, :, :])
    else:
        u = None
    if g is not None:
        g = matvec(g[:, None, :], basis)
        g += coord
        if c is not None:
            g.addcmul_(c, norm)
    return s, u, g


def curv_mode(curv, mode='max'):
    """Compute a specific curvature from its components

    Parameters
    ----------
    curv : (N, 2) tensor
        Curvature components
    mode : {'gaussian', 'mean', 'min', 'max', None}, default='max'
        'gaussian' : Gaussian curvature = s1*s2
        'mean' : Mean curvature = (s1 + s2) / 2
        'max' : Curvature with maximum absolute magnitude = s1
        'min' : Curvature with minimum absolute magnitude = s2
        None : Individual curvature components = s1, s2

    Returns
    -------
    curv : (N, [2]) tensor
        Curvature

    """
    if mode == 'gaussian':
        curv = curv.prod(-1)
    elif mode == 'mean':
        curv = curv.mean(-1)
    elif mode == 'max':
        curv = curv.gather(-1, curv.abs().argmax(-1, keepdim=True))[:, 0]
    elif mode == 'min':
        curv = curv.gather(-1, curv.abs().argmin(-1, keepdim=True))[:, 0]
    return curv


def vertex_curv(coord, faces, mode='max', return_direction=False,
                outliers=0.05, smooth=10):
    """Compute the curvature at each vertex

    Parameters
    ----------
    coord : (N, D) tensor
        Vertices coordinates
    faces : (M, K) tensor[long]
        Vertices indices of each face
    mode : {'gaussian', 'mean', 'min', 'max', None}, default='max'
        'gaussian' : Gaussian curvature = s1*s2
        'mean' : Mean curvature = (s1 + s2) / 2
        'max' : Curvature with maximum absolute magnitude = s1
        'min' : Curvature with minimum absolute magnitude = s2
        None : Individual curvature components = s1, s2
    return_direction : bool, default=False
        Return the principal directions
    outliers : float in (0 .. 0.5), default=0.05
        Clamp values outside this lower/upper quantile.
        0 = no outlier removal
        0.5 = all values clamped to the median
    smooth : int, default=10
        Number of smoothing (= averaging over the 1-ring) iterations

    Returns
    -------
    curv : (N, [2]) tensor
        Curvature
    dir : (N, 2, D) tensor, if `return_direction`
        Principal directions of curvature

    """
    s, u, _ = _vertex_curv(coord, faces, fixdf=2, dir=return_direction)
    s = curv_mode(s, mode)
    if outliers < 0 or outliers > 0.5:
        raise ValueError('outliers must be in (0 .. 0.5)')
    if outliers:
        s = quantile_clamp_(s, outliers, 1-outliers)
    if smooth:
        s = smooth_overlay(s, faces, smooth)
    return (s, u) if return_direction else s


def quantile_clamp_(overlay, qmin=0, qmax=1, alldim=False):
    """Clamp values outside specified quantiles (inplace)

    Parameters
    ----------
    overlay : (N, ...) tensor
    qmin : float in (0..1), default=0
    qmax : float in (0..1), default=1
    alldim : bool, default=False
        Pool all dimensions to compute quantiles.
        Otherwise, quantiles are computed across the first dimension

    Returns
    -------
    clamped_overlay : (N, ...) tensor

    """
    if qmin == 0 and qmax == 1:
        return overlay
    qmin, qmax = torch.quantile(overlay, torch.as_tensor([qmin, qmax]).to(overlay),
                                dim=0 if not alldim else None).unbind(-1)
    if alldim:
        overlay = overlay.clamp_(qmin, qmax)
    else:
        qmin = qmin.expand(overlay.shape)
        qmax = qmax.expand(overlay.shape)
        overlay[overlay < qmin] = qmin[overlay < qmin]
        overlay[overlay > qmax] = qmax[overlay > qmax]
    return overlay


def quantile_clamp(overlay, qmin=0, qmax=1, alldim=False):
    """Clamp values outside specified quantiles

    Parameters
    ----------
    overlay : (N, ...) tensor
    qmin : float in (0..1), default=0
    qmax : float in (0..1), default=1
    alldim : bool, default=False
        Pool all dimensions to compute quantiles.
        Otherwise, quantiles are computed across the first dimension

    Returns
    -------
    clamped_overlay : (N, ...) tensor

    """
    return quantile_clamp_(overlay.clone(), qmin, qmax, alldim)


def vertex_valence(faces, n):
    """Compute the valence (number of neighbors) of each vertex

    Parameters
    ----------
    faces : (M, K) tensor[long]
        Faces
    n : int
        Number of vertices

    Returns
    -------
    valence : (N,) tensor[long]

    """
    count = faces.new_zeros([n])
    for i in range(3):
        vertex_scatter_add_(count, faces[:, i], 2)
    count.floor_divide_(2)
    return count


def smooth_overlay(overlay, faces, nb_iter=1):
    """Smooth an overlay by averaging across the 1-ring

    Parameters
    ----------
    overlay : (N, *feat) tensor
    faces : (M, K) tensor[long]
    nb_iter : int, default=1

    Returns
    -------
    smooth_overlay : (N, *feat) tensor

    """
    overlay = overlay.clone()
    count = vertex_valence(faces, len(overlay)).mul_(2)
    for _ in range(overlay.ndim-1):
        count = count.unsqueeze(-1)

    for _ in range(nb_iter):
        neighbor_overlay = vertex_sample(overlay, faces)
        overlay *= 2
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                vertex_scatter_add_(overlay, faces[:, i], neighbor_overlay[:, j])
        overlay /= count
    return overlay


def adjacency(n, faces, dtype=None):
    """Compute the (sparse) adjacency matrix of a mesh

    Parameters
    ----------
    n : int
        Number of vertices
    faces : (M, K) tensor
        Faces
    dtype : torch.dtype
        Output data type. Default is `bool`.

    Returns
    -------
    adj : (n, n) sparse tensor[bool]
        Adjacency matrix

    """
    one = torch.ones([1], dtype=torch.uint8, device=faces.device)
    one = one.expand(len(faces))
    adj = torch.sparse_coo_tensor(faces[:, :2].T, one, [n, n])
    adj.add_(torch.sparse_coo_tensor(faces[:, 1:].T, one, [n, n]))
    adj.add_(torch.sparse_coo_tensor(faces[:, [0, 2]].T, one, [n, n]))
    adj.add_(adj.transpose(-1, -2))
    adj = adj.coalesce()
    dtype = dtype or torch.bool
    return adj.to(dtype)


def stiffness(vertices, faces):
    """Compute the (sparse) stiffness matrix 

    This function is only implemented for 3D meshes.

    Parameters
    ----------
    vertices : (N, 3) tensor
        Mesh vertices
    faces : (N, K) tensor
        Mesh faces

    Returns
    -------
    stiff : (N, N) sparse tensor
        Stiffness matrix

    References
    ----------
    1. "Discrete Laplace-Beltrami Operators for Shape Analysis and Segmentation"
       Reuter M, Biasotti S, Giorgi D, Patane G, Spagnuolo M
       Computers & Graphics (2009)
    """
    n = len(vertices)

    # NOTE
    #   . <u,v> = "angle formed by the vectors u and v"
    #   . cot(<u,v>) = dot(u, v) / |cross(u, v)|
    #   . |cross(u, v)| = 2 * area of the corresponding triangle
    #
    #   Let the triangle be             A
    #                                 /   \
    #                                B --- C
    #   For the weight assigned to edge B-C, I compute cot(<AB, AC>).
    #   Note that Martin Martin computes cot(<CA, AB>), but this 
    #   is equal to -cot(<AB, CA>) = cot(<AB, AC>). So everyting's fine.

    # compute triangles area (x 4)
    area4 = face_area(vertices, faces) * 4
    area4.clamp_min_(area4.mean() * 1e-4)

    triangles = vertex_sample(vertices, faces)

    # angle at vertex 0
    alpha0 = -dot(triangles[:,1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]) / area4
    a = torch.sparse_coo_tensor(faces[:, 1:].T, alpha0, [n, n])

    # angle at vertex 1
    alpha1 = -dot(triangles[:, 2] - triangles[:, 1], triangles[:, 0] - triangles[:, 1]) / area4
    a += torch.sparse_coo_tensor(faces[:, [0, 2]].T, alpha1, [n, n])

    # angle at vertex 2
    alpha2 = -dot(triangles[:, 0] - triangles[:, 2], triangles[:, 1] - triangles[:, 2]) / area4
    a += torch.sparse_coo_tensor(faces[:, :2].T, alpha2, [n, n])
    
    # make symmetric
    a += a.transpose(-1, -2)

    # diagonal
    a += torch.sparse_coo_tensor(faces[:, [0, 0]].T, -(alpha1+alpha2), [n, n])
    a += torch.sparse_coo_tensor(faces[:, [1, 1]].T, -(alpha0+alpha2), [n, n])
    a += torch.sparse_coo_tensor(faces[:, [2, 2]].T, -(alpha0+alpha1), [n, n])

    a = a.coalesce()
    return a


def face_cotangents(vertices, faces):
    """Compute cotangents in each face

    This function is only implemented for 3D meshes.

    Parameters
    ----------
    vertices : (N, 3) tensor
        Mesh vertices
    faces : (N, K) tensor
        Mesh faces

    Returns
    -------
    cot : (M, D) tensor
        Cotangents in each face

    References
    ----------
    1. "Discrete Laplace-Beltrami Operators for Shape Analysis and Segmentation"
       Reuter M, Biasotti S, Giorgi D, Patane G, Spagnuolo M
       Computers & Graphics (2009)
    """
    n = len(vertices)

    # compute triangles area (x 4)
    area4 = face_area(vertices, faces) * 4
    area4.clamp_min_(area4.mean() * 1e-4)

    triangles = vertex_sample(vertices, faces)

    # angle at vertex 0
    alpha0 = -dot(triangles[:,1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]) / area4

    # angle at vertex 1
    alpha1 = -dot(triangles[:, 2] - triangles[:, 1], triangles[:, 0] - triangles[:, 1]) / area4

    # angle at vertex 2
    alpha2 = -dot(triangles[:, 0] - triangles[:, 2], triangles[:, 1] - triangles[:, 2]) / area4
    
    return torch.stack([alpha0, alpha1, alpha2], dim=-1)


def stiffness_matvec(cot, faces, vector):
    """Apply the matrix-vector product between the stiffness matrix and a vector

    Parameters
    ----------
    cot : (M, D) tensor
        Cotangents in each face
    faces : (N, K) tensor
        Mesh faces
    vector : (N, 3) tensor
        Vector of vertices

    Returns
    -------
    matvec : (N, 3) tensor
        Matrix-vector product

    References
    ----------
    1. "Discrete Laplace-Beltrami Operators for Shape Analysis and Segmentation"
       Reuter M, Biasotti S, Giorgi D, Patane G, Spagnuolo M
       Computers & Graphics (2009)
    """
    triangles = vertex_sample(vector, faces)
    matvec = torch.zeros_like(vector)
    cot = cot.unsqueeze(-1)

    # offdiagonal
    vertex_scatter_add_(matvec, faces[:, 1], cot[:, 0] * triangles[:, 2])
    vertex_scatter_add_(matvec, faces[:, 2], cot[:, 0] * triangles[:, 1])
    vertex_scatter_add_(matvec, faces[:, 0], cot[:, 1] * triangles[:, 2])
    vertex_scatter_add_(matvec, faces[:, 2], cot[:, 1] * triangles[:, 0])
    vertex_scatter_add_(matvec, faces[:, 0], cot[:, 2] * triangles[:, 1])
    vertex_scatter_add_(matvec, faces[:, 1], cot[:, 2] * triangles[:, 0])

    # diagonal
    vertex_scatter_add_(matvec, faces[:, 0], - (cot[:, 1] + cot[:, 2]) * triangles[:, 0])
    vertex_scatter_add_(matvec, faces[:, 1], - (cot[:, 0] + cot[:, 2]) * triangles[:, 1])
    vertex_scatter_add_(matvec, faces[:, 2], - (cot[:, 0] + cot[:, 1]) * triangles[:, 2])

    return matvec


def mass(vertices, faces, lump=False):
    """Compute the (sparse) mass matrix 

    This function is only implemented for 3D meshes.

    Parameters
    ----------
    vertices : (N, 3) tensor
        Mesh vertices
    faces : (N, K) tensor
        Mesh faces
    lump : bool
        Return the majoriser diag(|B|1) instead of B
        (i.e., lump weights on the diagonal)

    Returns
    -------
    mass : (N, N) sparse tensor
        Mass matrix

    References
    ----------
    1. "Discrete Laplace-Beltrami Operators for Shape Analysis and Segmentation"
       Reuter M, Biasotti S, Giorgi D, Patane G, Spagnuolo M
       Computers & Graphics (2009)
    """
    n = len(vertices)

    area = face_area(vertices, faces)
    area /= 3 if lump else 12

    # diagonal
    a  = torch.sparse_coo_tensor(faces[:, [0, 0]].T, area, [n, n])
    a += torch.sparse_coo_tensor(faces[:, [1, 1]].T, area, [n, n])
    a += torch.sparse_coo_tensor(faces[:, [2, 2]].T, area, [n, n])

    if not lump:
        a += torch.sparse_coo_tensor(faces[:, 1:].T, area, [n, n])
        a += torch.sparse_coo_tensor(faces[:, [0, 2]].T, area, [n, n])
        a += torch.sparse_coo_tensor(faces[:, :2].T, area, [n, n])
        
        # make symmetric
        a += a.transpose(-1, -2)

    a = a.coalesce()
    return a


def mass_matvec(area, faces, vector, lump=False):
    """Apply the matrix-vector product between the stiffness matrix and a vector

    Parameters
    ----------
    area : (M, D) tensor
        Area of each face
    faces : (N, K) tensor
        Mesh faces
    vector : (N, 3) tensor
        Vector of vertices
    lump : bool
        Apply the majoriser diag(|B|1) instead of B
        (i.e., lump weights on the diagonal)

    Returns
    -------
    matvec : (N, 3) tensor
        Matrix-vector product

    References
    ----------
    1. "Discrete Laplace-Beltrami Operators for Shape Analysis and Segmentation"
       Reuter M, Biasotti S, Giorgi D, Patane G, Spagnuolo M
       Computers & Graphics (2009)
    """
    triangles = vertex_sample(vector, faces)
    matvec = torch.zeros_like(vector)

    area = area / (3 if lump else 6)
    area = area.unsqueeze(-1)

    # diagonal
    vertex_scatter_add_(matvec, faces[:, 0], area * triangles[:, 0])
    vertex_scatter_add_(matvec, faces[:, 1], area * triangles[:, 1])
    vertex_scatter_add_(matvec, faces[:, 2], area * triangles[:, 2])

    if not lump:
        # offdiagonal
        area /= 2
        vertex_scatter_add_(matvec, faces[:, 0], area * triangles[:, 1])
        vertex_scatter_add_(matvec, faces[:, 0], area * triangles[:, 2])
        vertex_scatter_add_(matvec, faces[:, 1], area * triangles[:, 0])
        vertex_scatter_add_(matvec, faces[:, 1], area * triangles[:, 2])
        vertex_scatter_add_(matvec, faces[:, 2], area * triangles[:, 0])
        vertex_scatter_add_(matvec, faces[:, 2], area * triangles[:, 1])

    return matvec



def mesh_area(vertices, faces):
    """Compute the integral area of the mesh"""
    return face_area(vertices, faces).sum()


def mesh_centroid_area(vertices, faces):
    """Compute the centroid and integral area of the mesh"""
    area = face_area(vertices, faces).unsqueeze(-1)
    centers = face_barycenter(vertices, faces)
    sumarea = area.sum(0)
    return (area * centers).sum(0) / sumarea, sumarea.squeeze(-1)


def mesh_centroid(vertices, faces):
    """Compute the centroid (center of mass) of the mesh"""
    return mesh_centroid_area(vertices, faces)[0]


def mesh_normalize_(vertices, faces):
    """Normalize to area and location of the unit sphere (inplace)"""
    center, area = mesh_centroid_area(vertices, faces)
    vertices -= center
    vertices /= area.div_(4*math.pi).sqrt_()
    return vertices


def mesh_normalize(vertices, faces):
    """Normalize to area and location of the unit sphere"""
    return mesh_normalize_(vertices.clone(), faces)
