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
    xyz = point[:,:3]
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
    def __init__(self, root,  npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.split = split

        self.normal_channel = normal_channel

        assert (split == 'train' or split == 'test')

        self.pairs = self._make_dataset(self.root, self.split)

        print('The size of {} data is {}'.format(split, len(self.pairs)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to a data sample

    def __len__(self):
        return len(self.pairs)

    def _get_item(self, index):
        if index in self.cache:
            tooth_points, jaw_points, label = self.cache[index]
        else:
            tooth_path = self.pairs[index][0]["tooth"]
            jaw_path = self.pairs[index][0]["jaw"]
            label = self.pairs[index][1]

            tooth_points = \
                np.loadtxt(tooth_path, delimiter=',').astype(np.float32)
            jaw_points = np.loadtxt(jaw_path, delimiter=',').astype(np.float32)

            if len(self.cache) < self.cache_size:
                self.cache[index] = (tooth_points, jaw_points, label)

        if self.uniform:
            tooth_points = farthest_point_sample(tooth_points, self.npoints)
            jaw_points = farthest_point_sample(jaw_points, self.npoints)
        else:
            if self.split == 'train':
                train_idx = np.array(range(tooth_points.shape[0]))
                tooth_points = tooth_points[train_idx[:self.npoints], :]
                train_idx = np.array(range(jaw_points.shape[0]))
                jaw_points = jaw_points[train_idx[:self.npoints], :]
            else:
                tooth_points = tooth_points[0:self.npoints, :]
                jaw_points = jaw_points[0:self.npoints, :]

        tooth_points[:, 0:3] = pc_normalize(tooth_points[:, 0:3])
        jaw_points[:, 0:3] = pc_normalize(jaw_points[:, 0:3])

        if not self.normal_channel:
            tooth_points = tooth_points[:, 0:3]
            jaw_points = jaw_points[:, 0:3]

        return tooth_points, jaw_points, label

    def __getitem__(self, index):
        return self._get_item(index)

    @staticmethod
    def _make_dataset(root, split):
        phase_dir = os.path.join(root, split)
        df = pd.read_csv(os.path.join(phase_dir, 'data.csv'))

        pairs = []
        for _, row in df.iterrows():
            jaw_path = os.path.join(phase_dir, row['jaw'])
            tooth_path = os.path.join(phase_dir, row['tooth'])
            label = row[2:].to_numpy().astype(np.float32)
            pairs.append(({
                'jaw': jaw_path,
                'tooth': tooth_path,
            }, label))

        return pairs


if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('./data/modelnet40_normal_resampled/',split='train', uniform=False, normal_channel=True,)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point,label in DataLoader:
        import ipdb; ipdb.set_trace()
        print(point.shape)
        print(label.shape)


