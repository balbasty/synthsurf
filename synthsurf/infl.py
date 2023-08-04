import torch
from .surf import (
    vertex_sample, 
    vertex_scatter, 
    vertex_scatter_add_, 
    vertex_valence,
    vertex_normal,
    face_area,
    face_normal,
    face_barycenter,
    face_cotangents,
    smooth_overlay,
    mesh_normalize,
    mesh_normalize_,
    stiffness,
    stiffness_matvec,
    mass,
    mass_matvec,
)
from .linalg import dot, matvec, spsolve
from .optim import ConjugateGradient


def energy_metric_distortion(vertices1, vertices0, faces):
    """Compute the energy corresponding to metric distortion in each vertex

    Parameters
    ----------
    vertices1 : (N, D) tensor
        Vertices coordinates in the deformed mesh
    vertices0 : (N, D) tensor
        Vertices coordinates in the original mesh
    faces : (M, D) tensor
        Mesh faces
    
    Returns
    -------
    energy : (N,) tensor
        Energy in each vertex
    """

    # extract the vertices in each triangle
    triangles1 = vertex_sample(vertices1, faces)
    triangles0 = vertex_sample(vertices0, faces)

    nv = len(vertices1)
    vertex_energy = vertices1.new_zeros([nv])

    edges = [[0, 1], [0, 2], [1, 2]]
    for i, j in edges:
        # compute the energy of each edge
        v0i = triangles0[:, i]
        v0j = triangles0[:, j]
        d0 = (v0i - v0j).square().sum(-1).sqrt()
        
        v1i = triangles1[:, i]
        v1j = triangles1[:, j]
        d1 = (v1i - v1j).square().sum(-1).sqrt()

        edge_energy = (d1 - d0).square()

        # accumulate into vertices
        vertex_scatter_add_(vertex_energy, faces[:, i], edge_energy)
        vertex_scatter_add_(vertex_energy, faces[:, j], edge_energy)

    # normalize
    vertex_energy /= vertex_valence(faces, nv)
    vertex_energy /= 2

    return vertex_energy


def grad_metric_distortion(vertices1, vertices0, faces):
    """Compute the energy corresponding to metric distortion in each vertex

    Parameters
    ----------
    vertices1 : (N, D) tensor
        Vertices coordinates in the deformed mesh
    vertices0 : (N, D) tensor
        Vertices coordinates in the original mesh
    faces : (M, D) tensor
        Mesh faces
    
    Returns
    -------
    energy : (N,) tensor
        Energy in each vertex
    gradient : (N,) tensor
        Derivative with respect to each vertex
    """

    # extract the vertices in each triangle
    triangles1 = vertex_sample(vertices1, faces)
    triangles0 = vertex_sample(vertices0, faces)

    nv = len(vertices1)
    vertex_energy = vertices1.new_zeros([nv])
    vertex_grad = torch.zeros_like(vertices1)

    edges = [[0, 1], [0, 2], [1, 2]]
    for i, j in edges:
        # compute the energy of each edge

        v0i = triangles0[:, i]
        v0j = triangles0[:, j]
        d0 = (v0i - v0j).square().sum(-1).sqrt()
        
        v1i = triangles1[:, i]
        v1j = triangles1[:, j]
        d1 = (v1i - v1j).square().sum(-1).sqrt()

        edge_energy = (d1 - d0).square()
        edge_gradient = (d1 - d0).unsqueeze(-1) * (v1j - v1i)

        # accumulate into vertices

        vertex_scatter_add_(vertex_energy, faces[:, i], edge_energy)
        vertex_scatter_add_(vertex_energy, faces[:, j], edge_energy)

        vertex_scatter_add_(vertex_grad, faces[:, i], edge_gradient)
        vertex_scatter_add_(vertex_grad, faces[:, j], -edge_gradient)

    # normalize
    valence = vertex_valence(faces, nv)
    vertex_grad /= valence.unsqueeze(-1)
    vertex_energy /= valence
    vertex_energy /= 2

    return vertex_energy, vertex_grad


def energy_oriented_area(vertices1, vertices0, faces):
    """Compute the energy correponding to oriented area
    
    This energy penalizes folds, as triangles inside folds have
    their normal antiparallel with the normal of the target shape
    (e.g. sphere).

    Parameters
    ----------
    vertices1 : (N, D) tensor
        Vertices coordinates in the deformed mesh
    vertices0 : (N, D) tensor
        Vertices coordinates in the original mesh
    faces : (M, D) tensor
        Mesh faces
    
    Returns
    -------
    energy : (M,) tensor
        Energy in each face

    """
    # I don't know what I am doing
    area1 = face_normal(vertices1, faces)
    sign1 = dot(area1, face_barycenter(vertices1, area1)).sign()
    area1 = area1.square().sum(-1).sqrt() * sign1
    area0 = face_area(vertices0, faces)
    mask = sign1 < 0

    energy = (area1 - area0).square()
    energy.masked_fill_(mask, 0)
    energy /= 2 * len(faces)
    return energy


def energy_spring(vertices, faces):
    """Compute the spring energy.

    Parameters
    ----------
    vertices : (N, D) tensor
        Vertices coordinates
    faces : (M, D) tensor
        Mesh faces
    
    Returns
    -------
    energy : (N,) tensor
        Energy in each vertex

    """

    # extract the vertices in each triangle
    triangles = vertex_sample(vertices, faces)

    nv = len(vertices)
    vertex_energy = vertices.new_zeros([nv])

    edges = [[0, 1], [0, 2], [1, 2]]
    for i, j in edges:
        # compute the energy of each edge
        vi = triangles[:, i]
        vj = triangles[:, j]
        edge_energy = (vi - vj).square().sum(-1)

        # accumulate into vertices
        vertex_scatter_add_(vertex_energy, faces[:, i], edge_energy)
        vertex_scatter_add_(vertex_energy, faces[:, j], edge_energy)

    # normalize
    vertex_energy /= vertex_valence(faces, nv)
    vertex_energy /= 2

    return vertex_energy


def grad_spring(vertices, faces):
    """Compute the spring energy.

    Parameters
    ----------
    vertices : (N, D) tensor
        Vertices coordinates
    faces : (M, D) tensor
        Mesh faces
    
    Returns
    -------
    energy : (N,) tensor
        Energy in each vertex
    gradient : (N,) tensor
        Derivative with respect to each vertex

    """

    # extract the vertices in each triangle
    triangles = vertex_sample(vertices, faces)

    nv = len(vertices)
    vertex_energy = vertices.new_zeros([nv])
    vertex_grad = torch.zeros_like(vertices)

    edges = [[0, 1], [0, 2], [1, 2]]
    for i, j in edges:
        # compute the energy of each edge
        vi = triangles[:, i]
        vj = triangles[:, j]
        edge_energy = (vi - vj).square().sum(-1)
        edge_grad = (vi - vj)

        # accumulate into vertices
        vertex_scatter_add_(vertex_energy, faces[:, i], edge_energy)
        vertex_scatter_add_(vertex_energy, faces[:, j], edge_energy)
        vertex_scatter_add_(vertex_grad, faces[:, i], edge_grad)
        vertex_scatter_add_(vertex_grad, faces[:, j], -edge_grad)

    # normalize
    valence = vertex_valence(faces, nv)
    vertex_grad /= valence.unsqueeze(-1)
    vertex_energy /= valence
    vertex_energy /= 2

    return vertex_energy, vertex_grad


def tangent_dist(vertices, faces):
    """Compute the average distance fromneighbors to the tangent plane
    
    Parameters
    ----------
    vertices : (N, D) tensor
        Vertices coordinates
    faces : (M, D) tensor
        Mesh faces

    Returns
    -------
    dist : (N,) tensor
    """
    triangles = vertex_sample(vertices, faces)
    norm = vertex_normal(vertices, faces)
    norm = vertex_sample(norm, faces)

    dist = vertices.new_zeros([len(vertices)])
    for i in range(3):
        d = 0
        for j in range(3):
            if i == j: continue
            p = triangles[:, j] - triangles[:, i]
            d += dot(p, norm[:, i]).abs_()
        vertex_scatter_add_(dist, faces[:, i], d)
    dist /= vertex_valence(faces, len(vertices))
    return dist


def inflate(vertices, faces, lr=0.1, threshold=1e-3, max_iter=1e3, armijo=1,
            smooth=(32, 16, 8, 4, 2, 1), alpha_spring=1, alpha_metric=1):
    """Inflate a surface

    Parameters
    ----------
    vertices : (N, D) tensor
        Vertices coordinates
    faces : (M, D) tensor
        Mesh faces
    lr : float
        Gradient descent learning rate
    threshold : float
        Threshold for acceptable "flateness".
        Flateness is measured as the normalized distance from each 
        vertex tangent plane to the vertex' neighbors.
    max_iter : int
        Maximum number of iterations
    smooth : int or list[int]
        Number of rings over which to smooth gradients.
        If a list, hierarchical optimization
    alpha_spring : float
        Weight for the spring term
    alpha_metric : float
        Weight for the metric distortion term

    Returns
    -------
    vertices : (N, D) tensor
        Vertices coordinates
    """

    vertices0, vertices = vertices, vertices.clone()
    armijo0 = armijo

    if isinstance(smooth, int):
        smooth = [smooth]

    for smo in smooth:

        armijo = armijo0
        for n in range(int(max_iter)):

            d = tangent_dist(vertices, faces).mean()
            if d < threshold:
                print('converged')
                break

            e_dist, g_dist = grad_metric_distortion(vertices, vertices0, faces)
            e_sprg, g_sprg = grad_spring(vertices, faces)
            e = alpha_metric * e_dist.mean() + alpha_spring * e_sprg.mean()
            g = alpha_metric * g_dist + alpha_spring * g_sprg
            if smo:
                g = smooth_overlay(g, faces, smo)

            print(n+1, e.item(), d.item(), armijo)

            if armijo0 == 0:
                # gradient descent
                vertices.sub_(g, alpha=lr)

            else:
                # Backtracking line search
                e0, armijo, armijo_prev, success = e, armijo, 0, False
                while armijo > 1e-5:
                    vertices.sub_(g, alpha=lr*(armijo - armijo_prev))
                    e_dist = energy_metric_distortion(vertices, vertices0, faces)
                    e_sprg = energy_spring(vertices, faces)
                    e = alpha_metric * e_dist.mean() + alpha_spring * e_sprg.mean()
                    if e < e0:
                        success = True
                        armijo *= 1.5
                        break
                    else:
                        armijo_prev = armijo
                        armijo /= 2

                if not success:
                    print('line search failed', armijo_prev)
                    vertices.add_(g, alpha=lr*armijo_prev)
                    break

    return vertices


def curv_inflate(vertices, faces, lr=1, tol=1e-13, max_iter=1000):
    """Inflate a surface by minimizing its curvature while preserving 
    its area. 

    Parameters
    ----------
    vertices : (N, D) tensor
    faces : (M, K) tensor
    lr : float
    tol : float
    max_iter : int

    Returns
    -------
    vertices : (N, D) tensor

    References
    ----------
    1. "Can Mean-Curvature Flow be Modified to be Non-singular?"
       Kazhdan M, Solomon J, Ben-Chen M
       Computer Graphics Forum (2012)
       https://doi.org/10.1111/j.1467-8659.2012.03179.x

    2. "LaPy: Toolbox for Differential Geometry on Triangle and Tetrahedra Meshes"
       Reuter M
       GitHub
       https://github.com/Deep-MI/LaPy
    """
    def mv(M, x):
        # Sparse matmul does not support batches
        y = torch.empty_like(x)
        for x1, y1 in zip(x, y):
            matvec(M, x1, out=y1)
        return y

    v = mesh_normalize(vertices, faces)
    A = stiffness(vertices, faces)
    for n in range(max_iter):
        v0 = v
        B = mass(v, faces, lump=True)
        Bv = mv(B, v.T).T
        v = spsolve(B + lr * A, Bv)
        v = mesh_normalize_(v, faces)

        diff = (v0 - v).T
        norm = dot(diff.flatten(), mv(B, diff).flatten())
        print(n, norm.item() / len(v))
        if norm < tol * len(v):
            break
    return v


def curv_inflate_matrixfree(vertices, faces, lr=1, tol=1e-13, max_iter=1000):
    """Inflate a surface by minimizing its curvature while preserving 
    its area. 

    Parameters
    ----------
    vertices : (N, D) tensor
    faces : (M, K) tensor
    lr : float
    tol : float
    max_iter : int

    Returns
    -------
    vertices : (N, D) tensor

    References
    ----------
    1. "Can Mean-Curvature Flow be Modified to be Non-singular?"
       Kazhdan M, Solomon J, Ben-Chen M
       Computer Graphics Forum (2012)
       https://doi.org/10.1111/j.1467-8659.2012.03179.x

    2. "LaPy: Toolbox for Differential Geometry on Triangle and Tetrahedra Meshes"
       Reuter M
       GitHub
       https://github.com/Deep-MI/LaPy
    """
    def solve_(v, F, b, max_iter=1024, tol=1e-8):
        solver = ConjugateGradient(
            forward=F,
            target=b,
            max_iter=max_iter, 
            tol=tol,
        )
        return solver.solve_(v)


    v = mesh_normalize(vertices, faces)
    cot = face_cotangents(vertices, faces)
    A = lambda u: stiffness_matvec(cot, faces, u)
    for n in range(max_iter):
        v0 = v.clone()
        area = face_area(v, faces)
        B = lambda u: mass_matvec(area, faces, u, lump=True)
        Bv = B(v)
        F  = lambda u: B(u) + lr * A(u)
        v = solve_(v, F, Bv, tol=0, max_iter=32)
        v = mesh_normalize_(v, faces)

        diff = (v0 - v)
        norm = dot(diff.flatten(), B(diff).flatten())
        print(n, norm.item() / len(v), end='\r')
        if norm < tol * len(v):
            break
    print('')
    return v
