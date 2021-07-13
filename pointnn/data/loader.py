""" Pytorch dataset implementation """
import torch
import math
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class ModelnetDataset(Dataset):
    def __init__(self, base, type='train', rotate=None, downsample=None,
                 classes=None, jitter=None):
        self.base_dir = Path(base)
        self.classes = classes or self._find_classes()
        self.downsample = downsample
        self.jitter = jitter
        self.type = type
        self.transform = RotateTransform(rotate) if rotate else None
        self.random_idxs = None
        self.jitter_pts = None
        self._find_clouds()
        self._cache_clouds()

    def shuffle_idxs(self):
        self.random_idxs = np.random.choice(self.point_count,
                                            size=self.downsample,
                                            replace=False)
        self.jitter_pts = None

    def _find_classes(self):
        dirs = (d for d in self.base_dir.iterdir() if d.is_dir())
        names = sorted([d.name for d in dirs])
        return names

    def _find_clouds(self):
        self.clouds = []
        for cl in self.classes:
            class_dir = self.base_dir / cl / self.type
            pcs = class_dir.glob('*.npy')
            self.clouds += [(p, cl) for p in pcs]

    def __len__(self):
        return len(self.clouds)

    def _cache_clouds(self):
        self.cache = {}
        self.point_count = None
        for path, _ in self.clouds:
            pc = np.load(path)
            if self.point_count is None:
                self.point_count = pc.shape[1]
            else:
                assert(self.point_count == pc.shape[1])
            self.cache[path] = pc

    def _get_cloud(self, idx):
        path, class_name = self.clouds[idx]
        idx = self.classes.index(class_name)
        pc = self.cache[path]
        class_ = torch.tensor(idx, dtype=torch.int64)
        all_points = torch.tensor(pc, dtype=torch.float)
        if self.transform:
            all_points = self.transform(all_points)
        return class_, all_points

    def __getitem__(self, idx):
        class_, points = self._get_cloud(idx)
        points = self._downsample(points, self.downsample)
        in_points = points
        target_points = points
        if self.jitter is not None:
            jitter_points = self._jitter(points)
            in_points = jitter_points
            if self.jitter == 'both':
                target_points = jitter_points
        return {'class': class_,
                'in_points': in_points,
                'target_points': target_points}

    @property
    def num_classes(self):
        return len(self.classes)

    def _downsample(self, pc, size):
        if size is None or pc.shape[1] == size:
            return pc
        if self.random_idxs is None:
            idxs = np.random.choice(pc.shape[1], size=size, replace=False)
        else:
            idxs = self.random_idxs
        ds_pc = pc[:, idxs]
        return ds_pc

    def _jitter(self, pc):
        f = 0.02
        if self.jitter_pts is None:
            self.jitter_pts = torch.empty_like(pc).normal_() * f
        return pc + self.jitter_pts


class RotateTransform:
    def __init__(self, type_='upright'):
        self.type = type_

    def __call__(self, points):
        n1, n2, n3 = torch.rand(3)
        angle = n1*2*math.pi
        s = torch.sin(angle)
        c = torch.cos(angle)
        rot_mat = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        if self.type == 'full':
            # Fast Random Rotation Matrices - James Arvo
            a2 = 2*math.pi*n2
            v = torch.tensor([[torch.cos(a2)*torch.sqrt(n3),
                              torch.sin(a2)*torch.sqrt(n3),
                              torch.sqrt(1-n3)]])
            h = torch.eye(3)-2*v*v.transpose(0, 1)
            rot_mat = torch.mm(-h, rot_mat)
        rot_points = torch.mm(rot_mat, points)
        return rot_points
