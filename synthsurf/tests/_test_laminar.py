from synthsurf.io import load_mesh, load_annot, load_label_volume, load_volume
from synthsurf.synth import synth_laminar_threshold, sample_laminar_surfaces
from synthsurf.surf import vertex_sample
from synthsurf.linalg import relabel, matvec
from jitfields.distance import mesh_sdt_extra, mesh_sdt
from jitfields.pushpull import pull
from jitfields.resize import resize
import glob
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import math


def rgb2hsv(x):
    R, G, B = x[..., :3].unbind(-1)
    V, iV = x[..., :3].max(-1)
    v, _ = x[..., :3].min(-1)
    C = V - v
    H = torch.where(iV == 0, ((G-B)/C) % 6,
        torch.where(iV == 1, ((B-R)/C) + 2,
                             ((R-G)/C) + 4))
    H *= 60
    S = torch.where(V == 0, x.new_zeros([]), C/V)
    H[~torch.isfinite(H)] = 0
    return torch.stack([H, S, V], -1)


def hsv2rgb(x):
    H, S, V = x[..., :3].unbind(-1)
    H /= 60
    C = V * S
    X = C * (1 - (H % 2 - 1).abs())
    Z = x.new_zeros([])
    R = torch.where((H < 1), C,
        torch.where((H < 2), X,
        torch.where((H < 4), Z,
        torch.where((H < 5), X,
                             C))))
    G = torch.where((H < 1), X,
        torch.where((H < 3), C,
        torch.where((H < 5), X,
                             Z)))
    B = torch.where((H < 2), Z,
        torch.where((H < 3), X,
        torch.where((H < 5), C,
                             X)))
    RGB = torch.stack([R, G, B], -1)
    RGB += (V * (1 - S)).unsqueeze(-1)
    return RGB


def load_lookup(fname):
    lookup = {}
    names = {}
    with open(fname, 'rt') as f:
        for row in f:
            row = row.split()
            label = int(row[0])
            lookup[label] = list(map(int, row[2:5]))
            names[label] = row[1]
    return lookup, names
            

SUB = 'sub-127989683000'
root = f'/autofs/space/pade_003/users/yb947/data/MRI/PSAMSEG/fs/{SUB}/surf'

mri, aff_mri = load_volume(f'{root}/../mri/orig.mgz', return_space=True) 

fname_psamseg = f'{root}/../../../{SUB}/seg_left.mgz'
psamseg, aff_psamseg = load_label_volume(fname_psamseg, return_space=True)
psamseg_lookup, psamseg_names = load_lookup(f'{root}/../../../{SUB}/lookup_table.txt')

fnames = list(sorted(glob.glob(f'{root}/lh.expanded*')))
fname_aparc = f'{root}/../label/lh.aparc.a2009s.annot'

vert, faces, aff, shape = load_mesh(fnames[0], return_space=True)
faces = faces.long()
vert = vert.float()

pial_vert, pial_faces = load_mesh(f'{root}/lh.pial')
pial_faces = pial_faces.long()
pial_vert = pial_vert.float()

aparc, ctab, names = load_annot(fname_aparc)
# aparc = relabel(aparc)
aparc.clamp_min_(0)  # what does -1 mean?
i = names.index('Medial_wall')
ctab[0, 0], ctab[0, 1], ctab[0, 2] = [255] * 3
ctab[i, 0], ctab[i, 1], ctab[i, 2] = [255] * 3

vert = [vert]
for fname in fnames[1:]:
    vert.append(load_mesh(fname)[0].float())
vert = torch.stack(vert, dim=-1)

# sample layers
tmap = synth_laminar_threshold(faces, 5, nb_iter=10)
layers = sample_laminar_surfaces(vert, tmap)

# concat white and pial
layers = [vert[:, :, :1], layers, vert[:, :, -1:]]
layers = torch.cat(layers, -1)

# rasterize

xrange = [layers[..., 0][:, 0].min().floor() - 5, layers[..., 0][:, 0].max().ceil() + 6]
yrange = [layers[..., 0][:, 1].min().floor() - 5, layers[..., 0][:, 1].max().ceil() + 6]
zrange = [layers[..., 0][:, 2].min().floor() - 5, layers[..., 0][:, 2].max().ceil() + 6]
x = (sum(xrange)/2).round()
y = (sum(yrange)/2).round() - 4
z = (sum(zrange)/2).round()
vx = xrange[1] - xrange[0]
vy = yrange[1] - yrange[0]
vz = zrange[1] - zrange[0]

coord = torch.meshgrid(torch.linspace(*xrange, (1024 * vx/vz).round().int()),
                       torch.arange(y, y+1),
                       torch.linspace(*zrange, 1024))
coord = torch.stack(coord, -1)
voxelsize = (vz - 1) / 1024

coord = coord.cuda()
layers = layers.cuda()
faces = faces.cuda()
aparc = aparc.cuda()
psamseg = psamseg.cuda()

labels_layer = torch.zeros([coord.shape[0], coord.shape[2]], dtype=torch.int32, device=coord.device)
labels_destrieux = torch.zeros([coord.shape[0], coord.shape[2]], dtype=torch.int32, device=coord.device)
colors = torch.zeros([coord.shape[0], coord.shape[2], 3], device=coord.device)


ids = [7, 6, 5, 4, 3, 2, 1]
aparcmax = aparc.max()
for layer, id in zip(reversed(layers.unbind(-1)), ids):
    sdt, nearest = mesh_sdt_extra(coord, layer, faces)
    sdt, nearest = sdt.squeeze(), nearest.squeeze()

    label1 = aparc[nearest]

    rgb = ctab[..., :3].to(label1.device) / 255
    hsv = rgb2hsv(rgb)
    hsv[..., 1] = hsv[..., 1] - (7 - id) * hsv[..., 1] / 7
    hsv[..., 2] = hsv[..., 2] - (id - 1) * hsv[..., 2] / 14
    rgb0 = hsv2rgb(hsv)
    rgb = vertex_sample(rgb0, label1)

    colors = torch.where(sdt.unsqueeze(-1) < 0, rgb, colors)
    labels_destrieux = torch.where(sdt < 0, label1, labels_destrieux)
    labels_layer.masked_fill_(sdt < 0, id)


mesh_to_psamseg = aff_psamseg.inverse() @ aff
mesh_to_psamseg = mesh_to_psamseg.to(coord)
coord_psamseg = matvec(mesh_to_psamseg[:3, :3], coord).add_(mesh_to_psamseg[:3, -1])
slice_psamseg = pull(psamseg.to(coord).unsqueeze(-1), coord_psamseg, order=0).squeeze()

for i in slice_psamseg.unique().tolist():
    if i == 0:
        continue
    rgb = psamseg_lookup[i]
    name = psamseg_names[i]
    if name.startswith('ctx') or (name.startswith('white_matter') and not name.endswith('hindbrain')):
        continue
    mask = slice_psamseg == i

    rgb = (torch.as_tensor(rgb) / 255).to(colors)
    colors = torch.where(mask.unsqueeze(-1), rgb, colors)

# cmap = mpl.colormaps['viridis'](torch.linspace(0, 1, labels.max()+1)).tolist()
# cmap0, *cmap = cmap
# random.shuffle(cmap)
# cmap = [cmap0, *cmap]
# ccmap = mpl.colors.LinearSegmentedColormap.from_list('shuffled', cmap, N=len(cmap))

# plt.imshow(labels.cpu(), cmap=ccmap, interpolation='nearest')
# plt.colorbar()
# plt.show(block=False)

# plt.figure()
# plt.imshow(colors.cpu(), interpolation='nearest')
# plt.show(block=False)

foo = 0

# make density maps

density = torch.zeros_like(slice_psamseg, dtype=torch.float32, device=colors.device)
size = torch.zeros_like(slice_psamseg, dtype=torch.float32, device=colors.device) # diameter

for i in slice_psamseg.unique().tolist():
    if i == 0:
        continue
    name = psamseg_names[i]
    if name.startswith('ctx') or (name.startswith('white_matter') and not name.endswith('hindbrain')):
        continue

    nbctrl = [random.randint(1, 8) for _ in range(2)]
    mndensity = random.random()
    mxdensity = random.random()
    mxdensity = (mndensity + mxdensity) / 2
    mndensity = mndensity / 2
    mndensity *= 4e4
    mxdensity *= 4e4

    density1 = torch.rand(nbctrl, device=density.device).mul_(mxdensity-mndensity).add_(mndensity)
    density1 = resize(density1, shape=density.shape, ndim=2, order=3, prefilter=False)
    density = torch.where(slice_psamseg == i, density1, density)

    nbctrl = [random.randint(1, 8) for _ in range(2)]
    mnsize = random.random()
    mxsize = random.random()
    mxsize = (mnsize + mxsize) / 2
    mnsize = mnsize / 2
    mnsize = mnsize * 30 + 5 
    mxsize = mxsize * 30 + 5

    size1 = torch.rand(nbctrl, device=size.device).mul_(mxsize-mnsize).add_(mnsize)
    size1 = resize(size1, shape=density.shape, ndim=2, order=3, prefilter=False)
    size = torch.where(slice_psamseg == i, size1, size)


for i in labels_layer.unique().tolist():
    if i == 0:
        continue
    for j in labels_destrieux.unique().tolist():
        if j == 0:
            continue

        mask = (labels_destrieux == j) & (labels_layer == i)

        nbctrl = [random.randint(1, 8) for _ in range(2)]
        mndensity = random.random()
        mxdensity = random.random()
        mxdensity = (mndensity + mxdensity) / 2
        mndensity = mndensity / 2
        mndensity *= 4e4
        mxdensity *= 4e4

        density1 = torch.rand(nbctrl, device=density.device).mul_(mxdensity-mndensity).add_(mndensity)
        density1 = resize(density1, shape=density.shape, ndim=2, order=3, prefilter=False)
        density = torch.where(mask, density1, density)

        nbctrl = [random.randint(1, 8) for _ in range(2)]
        mnsize = random.random()
        mxsize = random.random()
        mxsize = (mnsize + mxsize) / 2
        mnsize = mnsize / 2
        mnsize = mnsize * 30 + 5 
        mxsize = mxsize * 30 + 5
        
        size1 = torch.rand(nbctrl, device=size.device).mul_(mxsize-mnsize).add_(mnsize)
        size1 = resize(size1, shape=density.shape, ndim=2, order=3, prefilter=False)
        size = torch.where(mask, size1, size)


kernel = torch.as_tensor(
    [[0.25, 0.5, 0.25],
     [0.5, 1.0, 0.5],
     [0.25, 0.5, 0.25]]
).float()
kernel = kernel / kernel.sum()

density = torch.nn.functional.conv2d(density[None, None], kernel[None, None].to(density))[0, 0]
size = torch.nn.functional.conv2d(size[None, None], kernel[None, None].to(size))[0, 0]

stain_fraction = density * size.mul(0.5e-3).square() * (math.pi * 40e-3)
# stain_fraction.clamp_max_(1)

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(colors.cpu(), interpolation='nearest')
plt.colorbar()
plt.axis('off')
plt.subplot(2, 2, 2)
plt.imshow(stain_fraction.cpu(), interpolation='nearest')
plt.colorbar()
plt.axis('off')
plt.subplot(2, 2, 3)
plt.imshow(density.cpu(), interpolation='nearest')
plt.colorbar()
plt.axis('off')
plt.subplot(2, 2, 4)
plt.imshow(size.cpu(), interpolation='nearest')
plt.colorbar()
plt.axis('off')
plt.show(block=False)

foo = 0