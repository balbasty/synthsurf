import nibabel.freesurfer.io as fsio
import nibabel as nib
import torch
import numpy as np
import itertools


_np_to_torch_dtype = {
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex32,
    np.complex128: torch.complex64,
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    # upcast
    np.uint16: torch.int32,
    np.uint32: torch.int64,
    np.uint64: torch.int64,     # risk overflow
}

_torch_to_np_dtype = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.complex32: np.complex64,
    torch.complex64: np.complex128,
    torch.bool: np.bool_,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
}


if hasattr(np, 'complex256') and hasattr(torch, 'complex128'):
    _np_to_torch_dtype[np.complex256] = torch.complex128
    _torch_to_np_dtype[torch.complex128] = np.complex256


def to_np_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        dtype = _torch_to_np_dtype[dtype]
    dtype = np.dtype(dtype).type
    return dtype


def to_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype = _np_to_torch_dtype[to_np_dtype(dtype)]
    return dtype


def default_affine(shape, voxel_size=1, **backend):
    """Generate a RAS affine matrix

    Parameters
    ----------
    shape : list[int]
        Lattice shape
    voxel_size : [sequence of] float
        Voxel size
    dtype : torch.dtype, optional
    device : torch.device, optional

    Returns
    -------
    affine : (D+1, D+1) tensor
        Affine matrix

    """
    ndim = len(shape)
    aff = torch.eye(ndim+1, **backend)
    backend = dict(dtype=aff.dtype, device=aff.device)

    # set voxel size
    voxel_size = torch.as_tensor(voxel_size, **backend).flatten()
    pad = max(0, ndim - len(voxel_size))
    pad = [voxel_size[-1:]] * pad
    voxel_size = torch.cat([voxel_size, *pad])
    voxel_size = voxel_size[:ndim]
    aff[:-1, :-1] *= voxel_size[None, :]

    # set center fov
    shape = torch.as_tensor(shape, **backend)
    aff[:-1, -1] = -voxel_size * (shape - 1) / 2

    return aff


def load_label_volume(fname, return_space=False, numpy=False, dtype=None, device=None):
    """Load a volume of labels

    Parameters
    ----------
    fname : str
        Path to surface file

    return_space : bool, default=False
        Return the affine matrix of the volume

    numpy : bool, default=False
        Return numpy array instead of torch tensor

    Returns
    -------
    dat : (*spatial) tensor[integer]
        Volume of labels

    affine : (D+1, D+1) tensor, if `return_space`
        Orientation matrix (float64, cpu)

    """
    f = nib.load(fname)
    affine = f.affine
    d = np.asarray(f.dataobj)

    if not numpy:
        if not np.dtype(d.dtype).isnative:
            d = d.view(d.dtype.newbyteorder('=')).byteswap(inplace=True)
        d = torch.as_tensor(d, dtype=to_torch_dtype(dtype or d.dtype),
                            device=device)
        affine = torch.as_tensor(affine)
    else:
        d = d.astype(to_np_dtype(dtype))

    return (d, affine) if return_space else d


def load_volume(fname, return_space=False, numpy=False, dtype='float32', device=None):
    """Load a volume

    Parameters
    ----------
    fname : str
        Path to surface file

    return_space : bool, default=False
        Return the affine matrix of the volume

    numpy : bool, default=False
        Return numpy array instead of torch tensor

    dtype : str or torch.dtype or np.dtype
        Data type

    Returns
    -------
    dat : (*spatial, *channels) tensor[integer]
        Volume

    affine : (D+1, D+1) tensor, if `return_space`
        Orientation matrix (float64, cpu)

    """
    if isinstance(dtype, torch.dtype):
        dtype = _torch_to_np_dtype[dtype]

    f = nib.load(fname)
    affine = f.affine
    d = f.get_fdata(dtype=dtype)

    if not numpy:
        if not np.dtype(d.dtype).isnative:
            d = d.view(d.dtype.newbyteorder('=')).byteswap(inplace=True)
        d = torch.as_tensor(d, dtype=to_torch_dtype(dtype or d.dtype),
                            device=device)
        affine = torch.as_tensor(affine)
    else:
        d = d.astype(to_np_dtype(dtype))

    return (d, affine) if return_space else d


def _closest_orientation(lin):
    ndim = len(lin)
    aff0 = torch.eye(ndim, dtype=lin.dtype, device=lin.device)
    best_aff = None
    best_sse = float('inf')
    for perm in itertools.permutations(range(ndim)):
        for flip in itertools.product([-1, 1], repeat=ndim):
            aff = aff0[:, perm]
            aff = aff * torch.as_tensor(flip).to(aff)
            sse = (aff - lin).square().sum()
            if sse < best_sse:
                best_aff = aff
                best_sse = sse
    return best_aff


def load_mesh(fname, return_space=False, numpy=False):
    """Load a mesh in memory

    Parameters
    ----------
    fname : str
        Path to surface file

    return_space : bool, default=False
        Return the affine matrix and shape of the original volume

    numpy : bool, default=False
        Return numpy array instead of torch tensor

    Returns
    -------
    coord : (N, D) tensor
        Node coordinates.
        Each node has a coordinate in an ambient space.

    faces : (M, K) tensor
        Faces.
        Each face is made of K nodes, whose indices are stored in this tensor.
        For triangular meshes, K = 3.

    affine : (D+1, D+1) tensor, if `return_space`
        Mapping from the `coord`'s ambient space to a standard space.
        In Freesurfer surfaces, edges coordinates are also expressed in
        voxels of the original volumetric file, in which case the affine
        maps these voxel coordinates to millimetric RAS coordinates.

    shape : (D,) list[int], if `return_space`
        Shape of the original volume.

    """
    v, f, *meta = fsio.read_geometry(fname, read_metadata=return_space)

    if not numpy:
        if not np.dtype(v.dtype).isnative:
            v = v.view(v.dtype.newbyteorder('=')).byteswap(inplace=True)
        if not np.dtype(f.dtype).isnative:
            f = f.view(f.dtype.newbyteorder('=')).byteswap(inplace=True)
        v = torch.as_tensor(v, dtype=_np_to_torch_dtype[np.dtype(v.dtype).type])
        f = torch.as_tensor(f, dtype=_np_to_torch_dtype[np.dtype(f.dtype).type])

    if not return_space:
        return v, f

    meta = meta[0]
    aff = torch.eye(v.shape[-1] + 1)
    shape = None
    if 'volume' in meta and 'cras' in meta and 'voxelsize' in meta:
        shape = torch.as_tensor(meta['volume'])
        vx = torch.as_tensor(meta['voxelsize'], dtype=torch.float32)

        phys2ras = torch.eye(v.shape[-1] + 1)
        x, y, z, c = meta['xras'], meta['yras'], meta['zras'], meta['cras']
        x, y, z, c = torch.as_tensor(x), torch.as_tensor(y), torch.as_tensor(z), torch.as_tensor(c)
        phys2ras[:-1, :] = torch.stack([x, y, z, c], dim=1).to(phys2ras)

        orient2mesh = torch.eye(v.shape[-1] + 1)
        orient2mesh[:-1, :-1] = _closest_orientation(phys2ras[:-1, :-1])

        orient2phys = torch.eye(v.shape[-1] + 1)
        orient2phys.diagonal(0, -1, -2)[:-1] = vx.to(orient2phys)

        aff = phys2ras @ orient2phys @ orient2mesh.inverse()
        shape = shape.tolist()
    aff = aff.double()
    if numpy:
        aff = aff.numpy()

    return v, f, aff, shape


def load_overlay(fname, numpy=False):
    """Load an overlay (= map from vertex to scalar/vector value)

    Parameters
    ----------
    fname : str
        Path to overlay file

    numpy : bool, default=False
        Return numpy array instead of torch tensor

    Returns
    -------
    overlay : (N, [K]) tensor
        N is the number of vertices
        K is the dimension of the value space (or number of frames)

    """
    o = fsio.read_morph_data(fname)
    if not numpy:
        if not np.dtype(o.dtype).isnative:
            o = o.view(o.dtype.newbyteorder('=')).byteswap(inplace=True)
        o = torch.as_tensor(o, dtype=_np_to_torch_dtype[np.dtype(o.dtype).type])
    return o


def load_annot(fname, numpy=False):
    """Load an annotation (= map from vertex to label)

    Parameters
    ----------
    fname : str
        Path to overlay file

    numpy : bool, default=False
        Return numpy array instead of torch tensor

    Returns
    -------
    labels : (N,) tensor[integer]
        N is the number of vertices
    ctab : (K, 5) tensor[integer]
        K is the number of labels.
        5 is for RGBT + label id.
    names : (K,) list[str]
        The names of the labels.

    """
    a, c, n = fsio.read_annot(fname)
    n = [n1.decode('utf8') for n1 in n]

    if not numpy:
        if not np.dtype(a.dtype).isnative:
            a = a.view(a.dtype.newbyteorder('=')).byteswap(inplace=True)
        a = torch.as_tensor(a, dtype=_np_to_torch_dtype[np.dtype(a.dtype).type])
        if not np.dtype(c.dtype).isnative:
            c = c.view(c.dtype.newbyteorder('=')).byteswap(inplace=True)
        c = torch.as_tensor(c, dtype=_np_to_torch_dtype[np.dtype(c.dtype).type])

    return a, c, n
