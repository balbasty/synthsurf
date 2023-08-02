try:
    import plotly.graph_objects as go
except ImportError:
    go = None
from .surf import vertex_sample


def wireframe(vertices, faces, color='black', width=1, **prm):
    """Return a plotly object representing the mesh as a wireframe

    Parameters
    ----------
    vertices : (N, 3) tensor
    faces : (N, 3) tensor
    color : str or list[int]
    width : float

    Returns
    -------
    lines : go.Scatter3d
    """
    if not go:
        return None
    tri_points = vertex_sample(vertices, faces)
    Xe = []
    Ye = []
    Ze = []
    for T in tri_points:
        Xe.extend([T[k%3, 0] for k in range(4)]+[None])
        Ye.extend([T[k%3, 1] for k in range(4)]+[None])
        Ze.extend([T[k%3, 2] for k in range(4)]+[None])
    
    if isinstance(color, (list, tuple)):
        color = 'rgb({color[0]}, {color[1]}, {color[2]})'

    lines = go.Scatter3d(
        x=Xe,
        y=Ye,
        z=Ze,
        mode='lines',
        name='',
        line=dict(color=color, width=width, **prm),
    )  
    return lines


def surf(vertices, faces, **prm):
    """Return a plotly object representing the mesh as a surface

    Parameters
    ----------
    vertices : (N, 3) tensor
    faces : (N, 3) tensor
    color : str or list[int]
    width : float

    Returns
    -------
    lines : go.Scatter3d
    """
    mesh = go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], 
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        **prm,
    )
    return mesh


def show_wireframe(vertices, faces, color='black', width=1, **prm):
    """Plot the mesh as a wireframe in plotly

    Parameters
    ----------
    vertices : (N, 3) tensor
    faces : (N, 3) tensor
    color : str or list[int]
    width : float

    Returns
    -------
    fig : go.Figure
    """
    if not go:
        return None
    lines = wireframe(vertices, faces, color=color, width=width, **prm)
    fig = go.Figure(data=lines)
    fig.show()
    return fig