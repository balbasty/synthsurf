try:
    import plotly.graph_objects as go
    from plotly import colors
except ImportError:
    go = colors = None
import torch
from .surf import vertex_sample


def wireframe(vertices, faces, color='black', width=1, **prm):
    """Return a plotly object representing the mesh as a wireframe

    Parameters
    ----------
    vertices : (N, 3) tensor
    faces : (M, 3) tensor
    color : str or list[int]
    width : float

    Returns
    -------
    lines : go.Scatter3d
    """
    if not go:
        return None
    tri_points = vertex_sample(vertices, faces)
    tri_points = tri_points.detach().cpu()
    Xe = []
    Ye = []
    Ze = []
    for T in tri_points:
        Xe.extend([T[k % 3, 0] for k in range(4)]+[None])
        Ye.extend([T[k % 3, 1] for k in range(4)]+[None])
        Ze.extend([T[k % 3, 2] for k in range(4)]+[None])

    if isinstance(color, (list, tuple)):
        color = 'rgb({color[0]}, {color[1]}, {color[2]})'

    vertices = vertices.detach().cpu()
    faces = faces.detach().cpu()
    lines = go.Scatter3d(
        x=Xe,
        y=Ye,
        z=Ze,
        mode='lines',
        name='',
        line=dict(color=color, width=width, **prm),
    )
    return lines


def show_wireframe(vertices, faces, color='black', width=1, **prm):
    """Plot the mesh as a wireframe in plotly

    Parameters
    ----------
    vertices : (N, 3) tensor
    faces : (M, 3) tensor
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


def surf(vertices, faces, **prm):
    """Return a plotly object representing the mesh as a surface

    Parameters
    ----------
    vertices : (N, 3) tensor
    faces : (M, 3) tensor
    intensity : (N,) tensor, optional
    cmin, cmax : float, optional
    colorscale : str, optional

    Returns
    -------
    mesh : go.Mesh3d
    """
    vertices = vertices.detach().cpu()
    faces = faces.detach().cpu()
    if torch.is_tensor(prm.get('intensity', None)):
        prm['intensity'] = prm['intensity'].detach().cpu()
    mesh = go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        **prm,
    )
    return mesh


_wireframe = wireframe


def show_surf(vertices, faces, wireframe=False, **prm):
    """Plot the mesh as a surface in plotly

    Parameters
    ----------
    vertices : (N, 3) tensor
    faces : (M, 3) tensor
    intensity : (N,) tensor, optional
    cmin, cmax : float, optional
    colorscale : str, optional

    Returns
    -------
    fig : Figure
    data : list
    """
    if not go:
        return None
    data = [surf(vertices, faces, **prm)]
    if wireframe:
        if wireframe is not True:
            color = wireframe
        else:
            color = 'black'
        data += [_wireframe(vertices, faces, color=color)]
    fig = go.Figure(data=data)
    fig.show()
    return fig, data


def scalebar(colorscale, **prm):
    return go.Scatter3d(
        x=[0], y=[0], z=[0], mode='markers',
        marker=dict(
            size=0,
            color=[0],
            colorscale=colorscale,
            showscale=True,
            colorbar_thickness=23,
            **prm
        )
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def color_str2list(color: str):
    """
    Convert a string color (hexadecimal or rgb) into a list of values

    Parameters
    ----------
    color: str
        A color represented by a string:
        `"#ffffff"` or `"rgb(255,255,255)"`

    Returns
    -------
    color : list[float | int]
        A color represented by a list if values between 0 and 255:
        `[255, 255, 255]`
    """
    if color.startswith('#'):
        color = color[1:]
        mycolor = []
        while color:
            mycolor.append(int(color[:2], base=16))
            color = color[2:]
        return mycolor
    elif color.startswith('rgb'):
        color = color.split('(')[1].split(')')[0].split(',')
        color = list(map(float, map(str.strip, color)))
        return color
    else:
        return color


def color_list2str(color):
    """
    Convert a valued-color (`[255, 255, 255]`) into a rgb- or rgba- string
    (`rgb(255,255,255)`).

    Parameters
    ----------
    color : list[float | int]
        A color represented by a list if values between 0 and 255:
        `[255, 255, 255]`

    Returns
    -------
    color: str
        A color represented by a string:
        `"rgb(255,255,255)"` or `"rgb(255,255,255,1)"`
    """

    if len(color) == 3:
        color = [str(min(255, max(0, int(round(x))))) for x in color]
        return f'rgb({",".join(color)})'
    elif len(color) == 4:
        alpha = str(color[-1])
        color = [str(min(255, max(0, int(round(x))))) for x in color[:-1]]
        color += [alpha]
        return f'rgba({",".join(color)})'
    else:
        raise ValueError('Color should have 3 or 4 numbers')


def interpolate_colorscale(colorscale, i):
    """
    Interpolate a color scale at a given control point
    """
    j = -1
    ok = False
    while j < len(colorscale) - 1:
        j += 1
        if colorscale[j][0] >= i:
            ok = True
            break
    if not ok:
        return colorscale[-1][1]
    if colorscale[j][0] == i:
        if j < len(colorscale)-1:
            if colorscale[j+1][0] == i:
                return (colorscale[j][1], colorscale[j+1][1])
        return colorscale[j][1]
    else:
        w = (i - colorscale[j-1][0]) / (colorscale[j][0] - colorscale[j-1][0])
        if isinstance(colorscale[j][1], list):
            return [
                (1-w)*x + w * y
                for x, y in zip(colorscale[j-1][1], colorscale[j][1])
            ]
        else:
            x, y = colorscale[j-1][1], colorscale[j][1]
            return (1-w)*x + w * y


def make_transparent_colorscale(base: str, cmin, cmax, alpha=[]):
    """
    Make a transparent colorscale from a nontransparent one

    Parameters
    ----------
    base : str
        A plotly colorscale
    cmin : float
        The value corresponding to the bottom of the color scale
    cmax : float
        The value corresponding to the top of the color scale
    alpha : list[(float, float)]
        Control points for transparency

    Returns
    -------
    cmap
        New colormap
    """
    if isinstance(base, str):
        base = colors.get_colorscale(base)

    # Convert rgb string to list[int]
    base = [[i, color_str2list(rgb)] for i, rgb in base]

    # Convert intensities into control points
    alpha = [
        [(i-cmin)/(cmax-cmin), a]
        for (i, a) in alpha
    ]

    # Find all joint control points

    controls = set([x[0] for x in base]) | set([x[0] for x in alpha])
    controls = list(sorted(controls))

    # Build new color scale
    colorscale = []
    for control in controls:
        a = interpolate_colorscale(alpha, control)
        rgb = interpolate_colorscale(base, control)
        if isinstance(a, tuple):
            a = list(map(float, a))
            if isinstance(rgb, tuple):
                colorscale.append([control, [*rgb[0], a[0]]])
                colorscale.append([control, [*rgb[1], a[1]]])
            else:
                colorscale.append([control, [*rgb, a[0]]])
                colorscale.append([control, [*rgb, a[1]]])
        elif isinstance(rgb, tuple):
            colorscale.append([control, [*rgb[0], a]])
            colorscale.append([control, [*rgb[1], a]])
        else:
            a = float(a)
            colorscale.append([control, [*rgb, a]])

    # Convert list[int] to rgba string
    colorscale = [
        [i, color_list2str([min(255.0, max(0.0, x)) for x in rgba])]
        for i, rgba in colorscale
    ]
    return colorscale


def intensity_to_color(intensity, colorscale, cmin, cmax):
    """
    Map intensities to colors

    Parameters
    ----------
    intensity : (*batch) tensor
        Array of intensities
    colorscale : str or list[[int, str]]
        A colorscale (or its name)
    cmin : float
        Minimum value
    cmax : float
        Maximum value

    Returns
    -------
    color : (*batch, 4) tensor
        Color corresponding to each intensity
    """
    intensity = (intensity.float() - cmin) / (cmax - cmin)
    intensity.clamp_(0, 1)

    if isinstance(colorscale, str):
        colorscale = colors.get_colorscale(colorscale)

    colorscale = [
        [ctrl, color_str2list(rgb)]
        for ctrl, rgb in colorscale
    ]
    colorscale = [
        [ctrl, [x/255 for x in rgb[:3]] + rgb[3:]]
        for ctrl, rgb in colorscale
    ]

    color = torch.ones([len(intensity), 4])
    rgb = colorscale[0][1]
    color[:, :len(rgb)] = torch.as_tensor(rgb).to(color)

    mask = torch.zeros([len(intensity)], dtype=torch.bool)
    for j, (ctrl1, rgb1) in enumerate(colorscale):
        if j == 0:
            continue

        mask1 = intensity >= ctrl1
        if j < len(colorscale) - 1:
            ctrl2 = colorscale[j+1][0]
            mask1 &= intensity < ctrl2
        mask1 &= ~mask

        if not mask1.any():
            continue

        ctrl0 = colorscale[j-1][0]
        rgb0 = colorscale[j-1][1]

        if ctrl0 == ctrl1:
            rgb = (torch.as_tensor(rgb0) + torch.as_tensor(rgb1)) / 2
        else:
            w = (intensity[mask1] - ctrl0) / (ctrl1 - ctrl0)
            rgb = torch.stack([
                (1-w)*x + w * y
                for x, y in zip(rgb0, rgb1)
            ], -1)

        color[mask1, :rgb.shape[-1]] = rgb.to(color)
        mask |= mask1

    color.clamp_(0, 1)
    return color


def merge_colors(*colors):
    """
    Merge multiple color overlays

    Parameters
    ----------
    colors : (*batch, 4) tensor

    Returns
    -------
    color : (*batch, 4) tensor
    """
    color = torch.ones_like(colors[0])
    for color1 in colors:
        # Prepare alpha
        alpha = color1[..., -1:]
        # Merge
        color[:, :3] = (
            color[:, :3] * (1 - alpha) +
            color1[:, :3] * alpha
        )
        color[:, -1:] += color1[:, -1:]
    color.clamp_(0, 1)
    return color
