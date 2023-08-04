import torch


def matvec(mat, vec, out=None):
    """Matrix-vector product (supports broadcasting)

    Parameters
    ----------
    mat : (..., M, N) tensor
        Input matrix.
    vec : (..., N) tensor
        Input vector.
    out : (..., M) tensor, optional
        Placeholder for the output tensor.

    Returns
    -------
    mv : (..., M) tensor
        Matrix vector product of the inputs

    """
    mat = torch.as_tensor(mat)
    vec = torch.as_tensor(vec)[..., None]
    if out is not None:
        out = out[..., None]

    mv = torch.matmul(mat, vec, out=out)
    mv = mv[..., 0]
    if out is not None:
        out = out[..., 0]
    return mv


def dot(a, b, keepdim=False, out=None):
    """(Batched) dot product

    Parameters
    ----------
    a : (..., N) tensor
    b : (..., N) tensor
    keepdim : bool, default=False
    out : tensor, optional

    Returns
    -------
    ab : (..., [1]) tensor

    """
    a = a[..., None, :]
    b = b[..., :, None]
    ab = torch.matmul(a, b, out=out)
    if keepdim:
        ab = ab[..., 0]
    else:
        ab = ab[..., 0, 0]
    return ab


def outer(a, b, out=None):
    """Outer product of two (batched) tensors

    Parameters
    ----------
    a : (..., N) tensor
    b : (..., M) tensor
    out : (..., N, M) tensor, optional

    Returns
    -------
    out : (..., N, M) tensor

    """
    a = a.unsqueeze(-1)
    b = b.unsqueeze(-2)
    return torch.matmul(a, b)



if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'solve'):
    _solve_lu = torch.linalg.solve
else:
    _solve_lu = lambda A, b: torch.solve(b, A)[0]


def lmdiv(a, b, method='lu', rcond=1e-15, out=None):
    r"""Left matrix division ``inv(a) @ b``.

    Parameters
    ----------
    a : (..., m, n) tensor_like
        Left input ("the system")
    b : (..., m, k) tensor_like
        Right input ("the point")
    method : {'lu', 'chol', 'svd', 'pinv'}, default='lu'
        Inversion method:
        * 'lu'   : LU decomposition. ``a`` must be invertible.
        * 'chol' : Cholesky decomposition. ``a`` must be positive definite.
        * 'svd'  : Singular Value decomposition.
        * 'pinv' : Moore-Penrose pseudoinverse (by means of svd).
    rcond : float, default=1e-15
        Cutoff for small singular values when ``method == 'pinv'``.
    out : tensor, optional
        Output tensor (only used by methods 'lu' and 'chol').

    .. note:: if ``m != n``, the Moore-Penrose pseudoinverse is always used.

    Returns
    -------
    x : (..., n, k) tensor
        Solution of the linear system.

    """
    if a.shape[-1] != a.shape[-2]:
        method = 'pinv'
    if method.lower().startswith('lu'):
        # TODO: out keyword
        return _solve_lu(a, b)
    elif method.lower().startswith('chol'):
        u = torch.cholesky(a, upper=False)
        return torch.cholesky_solve(b, u, upper=False, out=out)
    elif method.lower().startswith('svd'):
        u, s, v = torch.svd(a)
        s = s[..., None]
        return v.matmul(u.transpose(-1, -2).matmul(b) / s)
    elif method.lower().startswith('pinv'):
        return torch.pinverse(a, rcond=rcond).matmul(b)
    else:
        raise ValueError('Unknown inversion method {}.'.format(method))


def isin(tensor, labels):
    """Returns a mask for elements that belong to labels

    Parameters
    ----------
    tensor : (*shape_tensor) tensor_like
        Input tensor
    labels : (*shape_labels, nb_labels) tensor_like
        Labels.
        `shape_labels` and `shape_tensor` should be broadcastable.

    Returns
    -------
    mask : (*shape) tensor[bool]

    """

    tensor = torch.as_tensor(tensor)
    if isinstance(labels, set):
        labels = list(labels)
    labels = torch.as_tensor(labels)

    if labels.shape[-1] == 1:
        # only one label in the list
        return tensor == labels[..., 0]

    mask = tensor.new_zeros(tensor.shape, dtype=torch.bool)
    for label in torch.unbind(labels, dim=-1):
        mask = mask | (tensor == label)

    return mask


def relabel(x, lookup=None):
    """Relabel a label tensor according to a lookup table

    Parameters
    ----------
    x : tensor[integer]
        Input tensor of labels
    lookup : sequence of [sequence of] int
        By default, relabel contiguously

    Returns
    -------
    x : tensor
        Relabeled tensor

    """
    if lookup is None:
        labelsinp = x.unique()
        labelmin = labelsinp.min()
        nbneg = (labelsinp < 0).sum()
        labelsinp -= labelmin
        labelsout = torch.arange(-nbneg, len(labelsinp)-nbneg, dtype=x.dtype, device=x.device)
        lookup = labelsinp.new_zeros(labelsinp.max()+1)
        lookup.scatter_(0, labelsinp.long(), labelsout)
        return lookup[x.long() - labelmin]

    if torch.is_tensor(lookup):
        lookup = lookup.tolist()
    out = torch.zeros_like(x)
    for i, j in enumerate(lookup):
        out[isin(x, j)] = i
    return out


def spsolve(A, b):
    """Sparse solve

    !!! warning
        A must be a single matrix, batching not supported.

    Parameters
    ----------
    A : (N, N) sparse tensor
    b : (..., N) tensor

    Returns
    -------
    x : (..., N) tensor
    """
    if b.is_cuda:
        from cupy.sparse.linalg import spsolve
        from cupy.sparse import coo_matrix, csr_matrix
        from jitfields.bindings.cuda.utils import to_cupy, from_cupy
        topy = to_cupy
        frompy = from_cupy
    else:
        from scipy.sparse.linalg import spsolve 
        from scipy.sparse import coo_matrix, csr_matrix
        topy = lambda x: x.numpy()
        frompy = torch.as_tensor

    A = A.coalesce()
    ind = topy(A.indices())
    val = topy(A.values())
    A = csr_matrix(coo_matrix((val, (ind[0], ind[1])), A.shape))
    b = topy(b)
    x = spsolve(A, b)
    return frompy(x)

