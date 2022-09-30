"""
Classification Model
Author: Wenxuan Wu
Date: September 2019
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear

from model.fpm import FeaturePropogationModule
from utils.pointconv_util import PointConvDensitySetAbstraction


class PointConvEncoder(nn.Module):

    def __init__(self, feature_dim=3, scale=1):
        super(PointConvEncoder, self).__init__()

        self.sa1 = PointConvDensitySetAbstraction(npoint=512 * scale, nsample=32 * scale, in_channel=feature_dim + 3, mlp=[64, 64, 128], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=128 * scale, nsample=64 * scale, in_channel=128 + 3, mlp=[128, 128, 256], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], bandwidth = 0.4, group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.7)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.7)

    def forward(self, xyz, feat):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, feat)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))

        return x


class PointConvDensityClsSsg(nn.Module):
    def __init__(self, nvalues, ntooth=28, sep_enc=False):
        super(PointConvDensityClsSsg, self).__init__()

        self.nvalues = nvalues
        self.ntooth = ntooth
        self.sep_enc = sep_enc

        if self.sep_enc:
            self.tooth_encoders = nn.ModuleList([
                PointConvEncoder()
                for _ in range(self.ntooth)
            ])
        else:
            self.tooth_encoder= PointConvEncoder()
        self.jaw_encoder = PointConvEncoder(scale=2)
        self.fpm = FeaturePropogationModule()
        self.fcs = nn.ModuleList([
            Linear(self.fpm.out_features, self.nvalues)
            for _ in range(self.ntooth)
        ])

    def forward(self, tooth_pcs, jaw_pc):
        if self.sep_enc:
            tooth_fea = [self.tooth_encoders[idx](
                tooth_pcs[:, idx, :3, :],
                tooth_pcs[:, idx, 3:, :],
            ) for idx in range(tooth_pcs.shape[1])]
            tooth_fea = torch.stack(tooth_fea, dim=1)
        else:
            tooth_pcs_re = tooth_pcs.reshape((
                -1,
                tooth_pcs.shape[2],
                tooth_pcs.shape[3],
            ))
            tooth_fea = self.tooth_encoder(
                tooth_pcs_re[:, :3, :],
                tooth_pcs_re[:, 3:, :],
            )
            tooth_fea = tooth_fea.reshape((
                tooth_pcs.shape[0],
                tooth_pcs.shape[1],
                tooth_fea.shape[-1],
            ))

        jaw_fea = self.jaw_encoder(jaw_pc[:, :3, :], jaw_pc[:, 3:, :])
        tooth_fea = torch.cat((tooth_fea, jaw_fea.unsqueeze(1)), dim=1)
        tooth_fea_fpm = self.fpm(tooth_fea)
        tooth_fea = tooth_fea + tooth_fea_fpm
        out = torch.stack([
            fc(tooth_fea[:, idx, :])
            for idx, fc in enumerate(self.fcs)
        ], dim=1)

        return out
