import sys
import os
sys.path.insert(0, f'{os.path.dirname(__file__)}/../../')

from synthsurf.ico import make_icosphere
from synthsurf.surf import vertex_sample

v, f = make_icosphere(4)


import plotly
import plotly.graph_objects as go

tri_points = vertex_sample(v, f)
Xe = []
Ye = []
Ze = []
for T in tri_points:
    Xe.extend([T[k%3, 0] for k in range(4)]+[None])
    Ye.extend([T[k%3, 1] for k in range(4)]+[None])
    Ze.extend([T[k%3, 2] for k in range(4)]+[None])
       
#define the trace for triangle sides
lines = go.Scatter3d(
    x=Xe,
    y=Ye,
    z=Ze,
    mode='lines',
    name='',
    line=dict(color='rgb(70,70,70)', width=1)
)  

mesh = go.Mesh3d(
    x=v[:, 0], y=v[:, 1], z=v[:, 2], 
    i=f[:, 0], j=f[:, 1], k=f[:, 2],
)

fig = go.Figure(data=[mesh, lines])

fig.show()

foo = 0

