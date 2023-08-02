try:
    import matplotlib.pyplot as plt
    import matplotlib
except ImportError:
    plt = None
    matplotlib = None
import torch
from .surf import vertex_sample


def surf(vertices, faces, vcolor=None, fcolor=None, ax=None, **kwargs):
    """Return a 3D axis containing the mesh as a surface

    Parameters
    ----------
    vertices : (N, 3) tensor
    faces : (M, 3) tensor
    vcolor : (N,) tensor
        Intensity at each vertex
    fcolor : (M,) tensor
        Intensity at each face
    
    Returns
    -------
    ax : plt.Axis

    """
    if not plt:
        return None

    if ax is None:
        ax = plt.gcf()
    if isinstance(ax, matplotlib.figure.Figure):
        ax = ax.add_subplot(projection='3d')

    if fcolor is None and vcolor is not None:
        fcolor = vertex_sample(vcolor, faces)
        fcolor = fcolor.mean(1)
    
    vertices = vertices.detach().cpu()
    faces = faces.detach().cpu()
    trisurf = ax.plot_trisurf(
        vertices[:, 0], 
        vertices[:, 1], 
        faces,
        vertices[:, 2], 
        linewidth=0.2, 
        antialiased=True,
        **kwargs,
    )

    if fcolor is not None:
        if torch.is_tensor(fcolor):
            fcolor = fcolor.detach().cpu()
        trisurf.set_array(fcolor)

    return ax