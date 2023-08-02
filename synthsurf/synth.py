from jitfields.pushpull import pull
from .surf import smooth_overlay
import torch


def synth_smooth_flow(shape, resolution, dmax=0.1, dunit='%', **backend):
    """Synthesize a smooth flow field

    Parameters
    ----------
    shape : list[int]
        Number of nodes
    resolution : float or list[float]
        Node spacing, in mm
    dmax : float
        Maximum displacement
    dunit : {'%', 'mm', 'vox'}
        `dmax` unit, where '%' means percentage of the field of view
    
    Returns
    -------
    flow : (*shape, D)
        Spline coefficients encoding a displacement field
    affine : (D+1, D+1)
        Orientation matrix of the flow field
    """
    backend.setdefault('dtype', torch.get_default_dtype())
    backend.setdefault('device', torch.device('cpu'))

    # make affine
    ndim = len(shape)
    affine = torch.eye(ndim, **backend)
    resolution = torch.as_tensor(resolution, **backend)
    affine.diagonal(0, -2, -1)[:-1] = resolution
    affine[:-1, -1] = -0.5 * torch.as_tensor(shape, **backend).sub_(1) * resolution

    # convert dmax to voxels
    dmax = torch.as_tensor(dmax, **backend)
    if dunit == '%':
        dmax = torch.as_tensor(shape, **backend) * dmax
    elif dunit == 'mm':
        dmax = dmax / resolution

    # make flow
    flow = torch.rand([*shape, ndim], **backend).sub_(0.5).mul_(2*dmax)

    return flow, affine


def synth_smooth_overlay(faces, nb_channels=None, vmin=0, vmax=1, nb_iter=100, dtype=None):
    """Synthesize a random overlay map that is smooth along the surface

    Parameters
    ----------
    faces : (M, D) tensor[integer]
        Mesh faces
    nb_channels : int
        Number of channels in the overlay
        (default: 1, and there is not explicit channel dimension)
    vmin : float
        Minimum value of the overlay
    vmax : float
        Maximum value of the overlay
    nb_iter : int
        Number of smoothing iterations along the 1-ring neighbor
    dtype : torch.dtype
        Output data type
    
    Returns
    -------
    overlay : (nb_vertices, [nb_channels]) tensor
        Output overlay

    """
    dtype = dtype or torch.get_default_dtype()
    N = faces.max() + 1
    has_channels = bool(nb_channels)
    C = nb_channels or 1

    overlay = torch.rand([N, C], device=faces.device, dtype=dtype or torch.get_default)
    overlay.mul_(vmax - vmin).add_(vmin)
    overlay = smooth_overlay(overlay, faces, nb_iter)

    if not has_channels:
        overlay = overlay.squeeze(-1)
    return overlay


def synth_laminar_threshold(faces, nb_thresholds=None, nb_iter=100, dtype=None):
    """Synthesize a random threshold map that is smooth along the surface

    Parameters
    ----------
    faces : (M, D) tensor[integer]
        Mesh faces
    nb_thresholds : int
        Number of increasing thresholds
        (default: 1, and there is not explicit channel dimension)
    nb_iter : int
        Number of smoothing iterations along the 1-ring neighbor
    dtype : torch.dtype
        Output data type
    
    Returns
    -------
    threshold : (nb_vertices, [nb_thresholds]) tensor
        Output threshold, in (0, 1).
        If nb_thresholds > 1, they are monotonically increasing (and bounded by 1).

    """
    has_channels = bool(nb_thresholds)
    C = nb_thresholds or 1

    thresholds = synth_smooth_overlay(faces, C+1, nb_iter=nb_iter, dtype=dtype)
    thresholds = thresholds.cumsum(dim=-1)
    thresholds = thresholds[:, :-1] / thresholds[:, -1:]

    if not has_channels:
        thresholds = thresholds.squeeze(-1)
    return thresholds


def sample_laminar_surfaces(vertices, thresholds, order=3, prefilter=True):
    """Sample laminar surfaces at specific thresholds along the cortical depth

    Parameters
    ----------
    vertices : (N, D, K) tensor
        Expanding vertices
    thresholds : (N, [C])
        Laminar thresholds
    order : int
        Spline interpolation order along the expansion dimension
    prefilter : bool
        Whether to spline-prefilter the vertices 
        (only use if `vertices` does not already contain 1D spline coefficients)

    Returns
    -------
    vertices : (N, D, [C]) tensor
        Sampled surfaces
    """
    N, D, K = vertices.shape

    pullopt = dict(order=order, bound='dct2', prefilter=prefilter)

    if thresholds.ndim == 1:
        thresholds = thresholds.unsqueeze(-1)
        has_channels = False
    else:
        has_channels = True

    thresholds = thresholds * (K-1)
    vertices = vertices.movedim(1, -1)                      # [N, K, D]
    thresholds = thresholds.unsqueeze(-1)                   # [N, C, 1]
    vertices = pull(vertices, thresholds, **pullopt)        # [N, C, D]
    vertices = vertices.movedim(1, -1)                      # [N, D, C]

    if not has_channels:
        vertices = vertices.squeeze(-1)
    return vertices