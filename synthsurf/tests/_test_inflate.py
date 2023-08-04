import sys
import os
sys.path.insert(0, f'{os.path.dirname(__file__)}/../../')

from synthsurf.io import load_mesh
from synthsurf.infl import inflate, curv_inflate, curv_inflate_matrixfree
from synthsurf.plotly import show_surf, show_wireframe

SUB = 'sub-127989683000'
root = f'/autofs/space/pade_003/users/yb947/data/MRI/PSAMSEG/fs/{SUB}/surf'

v, f = load_mesh(f'{root}/lh.white')
f = f.long()
v = v.float()

v = v.cuda()
f = f.cuda()
# v1 = curv_inflate(v, f, max_iter=6)
v1 = curv_inflate_matrixfree(v, f, max_iter=512)

show_surf(v, f)
show_surf(v1, f)

foo = 0