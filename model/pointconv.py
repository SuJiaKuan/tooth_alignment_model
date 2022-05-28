"""
Classification Model
Author: Wenxuan Wu
Date: September 2019
"""
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from utils.pointconv_util import PointConvDensitySetAbstraction


class PointConvEncoder(nn.Module):

    def __init__(self, feature_dim=3):
        super(PointConvEncoder, self).__init__()

        self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=feature_dim + 3, mlp=[64, 64, 128], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], bandwidth = 0.2, group_all=False)
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
    def __init__(self, nvalues=7):
        super(PointConvDensityClsSsg, self).__init__()

        self.tooth_encoder = PointConvEncoder()
        self.jaw_encoder = PointConvEncoder()
        self.fc = nn.Linear(2 * 256, nvalues)

    def forward(self, tooth_points, jaw_points):
        tooth_fea = self.tooth_encoder(
            tooth_points[:, :3, :],
            tooth_points[:, 3:, :],
        )
        jaw_fea = self.jaw_encoder(
            jaw_points[:, :3, :],
            jaw_points[:, 3:, :],
        )
        fea = torch.cat((jaw_fea, tooth_fea), dim=1)
        out = self.fc(fea)

        return out

if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((8,3,2048))
    label = torch.randn(8,16)
    model = PointConvDensityClsSsg(num_classes=40)
    output= model(input)
    print(output.size())

