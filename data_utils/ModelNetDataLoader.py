import numpy as np
import warnings
import os

import pandas as pd
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):

    def __init__(
        self,
        root,
        values,
        npoint=1024,
        ntooth=28,
        split='train',
        uniform=False,
        normal_channel=True,
        cache_size=15000,
    ):
        self.root = root
        self.values = values
        self.npoints = npoint
        self.ntooth = ntooth
        self.uniform = uniform
        self.split = split

        self.normal_channel = normal_channel

        assert (split == 'train' or split == 'test')

        self.items = self._make_dataset(
            self.root,
            self.split,
            self.ntooth,
            self.values,
        )

        print('The size of {} data is {}'.format(split, len(self.items)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to a data sample

    def __len__(self):
        return len(self.items)

    def _get_item(self, index):
        if index in self.cache:
            tooth_pcs, labels = self.cache[index]
        else:
            tooth_infos = self.items[index]
            tooth_pcs = []
            labels = []
            for tooth, filepath, label in tooth_infos:
                tooth_pcs.append(
                    np.loadtxt(filepath, delimiter=',').astype(np.float32)
                )
                labels.append(label)

            if len(self.cache) < self.cache_size:
                self.cache[index] = (tooth_pcs, labels)

        tooth_pcs_ = []
        for tooth_pc in tooth_pcs:
            if self.uniform:
                tooth_pc_ = farthest_point_sample(tooth_pc, self.npoints)
            else:
                if self.split == 'train':
                    train_idx = np.random.choice(
                        tooth_pc.shape[0],
                        size=min(tooth_pc.shape[0], self.npoints),
                        replace=False,
                    )
                    tooth_pc_ = tooth_pc[train_idx, :]
                else:
                    tooth_pc_ = tooth_pc[0:self.npoints, :]

            # tooth_pc_[:, 0:3] = pc_normalize(tooth_pc_[:, 0:3])

            if not self.normal_channel:
                tooth_pc_ = tooth_pc_[:, 0:3]

            tooth_pcs_.append(tooth_pc_)

        jaw_pc = np.vstack(tooth_pcs)
        jaw_npoints = int((self.npoints * self.ntooth) / 7)
        if self.uniform:
            jaw_pc_ = farthest_point_sample(jaw_pc, jaw_npoints)
        else:
            if self.split == 'train':
                train_idx = np.random.choice(
                    jaw_pc.shape[0],
                    size=min(jaw_pc.shape[0], jaw_npoints),
                    replace=False,
                )
                jaw_pc_ = jaw_pc[train_idx, :]
            else:
                tooth_npoints_lst = [int(jaw_npoints / self.ntooth)] * self.ntooth
                tooth_npoints_lst[-1] += jaw_npoints % self.ntooth
                jaw_pc_ = np.vstack([
                    tooth_pc[0:tooth_npoints, :]
                    for tooth_pc, tooth_npoints
                    in zip(tooth_pcs, tooth_npoints_lst)
                ])
        if not self.normal_channel:
            jaw_pc_ = jaw_pc_[:, 0:3]

        return np.array(tooth_pcs_), np.array(jaw_pc_), np.array(labels)

    def __getitem__(self, index):
        return self._get_item(index)

    @staticmethod
    def _make_dataset(root, split, ntooth, values):
        phase_dir = os.path.join(root, split)
        df = pd.read_csv(os.path.join(phase_dir, 'data.csv'), dtype={
            "name": str,
            "tooth": str,
        })

        items = []
        names = df['name'].unique()
        for name in names:
            df_name = df[df['name'] == name].iloc[:ntooth]
            tooth_infos = []
            for _, row in df_name.iterrows():
                tooth = row['tooth']
                filepath = os.path.join(
                    phase_dir,
                    name, "{}.txt".format(tooth),
                )
                row_values = row[2:].to_numpy().astype(np.float32)
                label = row_values[values]

                tooth_infos.append((tooth, filepath, label))

            items.append(tooth_infos)

        return items
