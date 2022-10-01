import argparse
import os

import torch
import numpy as np

from model.pointconv import PointConvDensityClsSsg as PointConvClsSsg
from data_utils.ModelNetDataLoader import farthest_point_sample
from utils.const import TEETH


def parse_args():
    parser = argparse.ArgumentParser("Movements Prediction")

    parser.add_argument(
        "data",
        type=str,
        help="Path to teeth directory to be predicted",
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--values",
        type=int,
        nargs="+",
        default=[0, 2, 3, 4, 5, 6],
        choices=list(range(7)),
        help="target values",
    )
    parser.add_argument(
        "--npoints",
        type=int,
        default=1024,
        help="Number of points per tooth",
    )
    parser.add_argument(
        "--normal",
        action="store_true",
        default=False,
        help="Whether to use normal information",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="pred.npy",
        help="Path to saved output prediciton",
    )

    return parser.parse_args()


def load_data(data_dir, npoints, normal_channel, ntooth=28, uniform=False):
    tooth_pcs = []
    for tooth in TEETH:
        filepath = os.path.join(data_dir, "{}.txt".format(tooth))
        tooth_pcs.append(
            np.loadtxt(filepath, delimiter=",").astype(np.float32)
        )

    tooth_pcs_ = []
    for tooth_pc in tooth_pcs:
        if uniform:
            tooth_pc_ = farthest_point_sample(tooth_pc, npoints)
        else:
            tooth_pc_ = tooth_pc[0:npoints, :]

        # tooth_pc_[:, 0:3] = pc_normalize(tooth_pc_[:, 0:3])

        if not normal_channel:
            tooth_pc_ = tooth_pc_[:, 0:3]

        tooth_pcs_.append(tooth_pc_)

    jaw_pc = np.vstack(tooth_pcs)
    jaw_npoints = int((npoints * ntooth) / 7)
    if uniform:
        jaw_pc_ = farthest_point_sample(jaw_pc, jaw_npoints)
    else:
        tooth_npoints_lst = [int(jaw_npoints / ntooth)] * ntooth
        tooth_npoints_lst[-1] += jaw_npoints % ntooth
        jaw_pc_ = np.vstack([
            tooth_pc[0:tooth_npoints, :]
            for tooth_pc, tooth_npoints
            in zip(tooth_pcs, tooth_npoints_lst)
        ])
    if not normal_channel:
        jaw_pc_ = jaw_pc_[:, 0:3]

    return torch.Tensor(tooth_pcs_).cuda(), torch.Tensor(jaw_pc_).cuda()


def load_regressor(checkpoint_path, values):
    regressor = PointConvClsSsg(len(values)).cuda()
    checkpoint = torch.load(checkpoint_path)
    regressor.load_state_dict(checkpoint["model_state_dict"])

    return regressor.eval()


def main(args):
    tooth_pcs, jaw_pc = load_data(args.data, args.npoints, args.normal)
    regressor = load_regressor(args.checkpoint, args.values)

    with torch.no_grad():
        tooth_pcs = torch.unsqueeze(tooth_pcs, 0).transpose(3, 2)
        jaw_pc = torch.unsqueeze(jaw_pc, 0).transpose(2, 1)
        pred = regressor(tooth_pcs, jaw_pc)[0]

    pred = pred.detach().cpu().numpy()

    np.save(args.output, pred)

    print(pred)
    print(pred.shape)
    print("Prediction saved in : {}".format(args.output))


if __name__ == '__main__':
    main(parse_args())
