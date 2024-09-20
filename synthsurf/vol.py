from jitfields.pushpull import pull
from .linalg import matvec


def sample_volume(vol, coord, affine=None, order=1, prefilter=True, bound='mirror'):
    """
    Sample values in a volume at a batch of coordinates

    Parameters
    ----------
    vol : (*spatial, *channels) tensor
        volume to sample
    coord : (*batch, ndim) tensor
        Coordinates at which to sample.
        If `affine=None`, assumed to be voxel coordinates.
        Else, assumed to be world coordinates.
    affine : (ndim+1, ndim+1) tensor
        Voxel-to-world orientation matrix of the volume.
    order : int
        Interpolation order
    prefilter : bool
        Whether to apply a spline-prefilter.
        If True, the volume is properly interpolated.
        If False, the volume is assumed to contain spline coefficients.
        If `order<=1`, it does not matter.
    bound : str
        Boundary conditions when sampling out-of-bounds

    Returns
    -------
    overlay : (*batch, *channels) tensor
    """
    ndim = coord.shape[-1]
    nchn = vol.ndim - ndim

    # Convert to voxels
    if affine is not None:
        affine = affine.inv()
        coord = matvec(affine[:ndim, :ndim], coord[..., None])[..., 0]
        coord += affine[:ndim, -1]

    # Insert singleton dimensions
    for _ in range(ndim):
        coord = coord[..., None, :]

    # Sampled
    overlay = pull(vol, coord, order=order, bound=bound, prefilter=prefilter)

    # Remove singleton dimensions
    channels = (slice(None),) * nchn
    for _ in range(ndim):
        overlay = overlay[(Ellipsis, None, *channels)]

    return overlay
