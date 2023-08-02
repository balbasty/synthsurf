import sys
import os
sys.path.insert(0, f'{os.path.dirname(__file__)}/../../')

from synthsurf.ico import make_icosphere
from synthsurf.surf import vertex_sample

v, f = make_icosphere(4)

from synthsurf.plot import surf

ax = surf(v, f)
ax.get_figure().show()

foo = 0

