import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import datetime
import logging
from pathlib import Path
from model.pointconv import PointConvDensityClsSsg as PointConvClsSsg
from utils.utils import test


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointConv')
    parser.add_argument('data_path', type=str, help='path to dataset')
    parser.add_argument('checkpoint', type=str, help='checkpoint')
    parser.add_argument('auc_max_thresh', type=int, help='maximum threshold for AUC calculation')
    parser.add_argument('--values', type=int, nargs='+', default=[0, 2, 3, 4, 5, 6], choices=list(range(7)), help='target values')
    parser.add_argument('--batchsize', type=int, default=8, help='batch size')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--num_workers', type=int, default=4, help='Worker Number [default: 4]')
    parser.add_argument('--model_name', default='pointconv', help='model name')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--tooth_wise', action='store_true', default=False, help='Whether to use tooth-wise metric calculation')
    return parser.parse_args()


def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = Path('./eval_experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%s_ModelNet40-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(file_dir, 'logs.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())
    logger.info('---------------------------------------------------EVAL---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    '''DATA LOADING'''
    logger.info('Load dataset ...')

    test_dataset = ModelNetDataLoader(args.data_path, args.values, npoint=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers)
    logger.info("The number of test data is: %d", len(test_dataset))

    seed = 3
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    '''MODEL LOADING'''
    logger.info('Load CheckPoint')
    classifier = PointConvClsSsg(len(args.values)).cuda()
    checkpoint = torch.load(args.checkpoint)
    classifier.load_state_dict(checkpoint['model_state_dict'])

    '''EVAL'''
    logger.info('Start evaluating...')

    classifier = classifier.eval()
    test_metric = test(
        classifier,
        testDataLoader,
        auc_max_thresh=args.auc_max_thresh,
        auc_curve_path=os.path.join(file_dir, "auc_curve.png"),
        tooth_wise=args.tooth_wise,
    )

    logger.info('Evaluation Metrics:')
    logger.info(json.dumps(test_metric, indent=2))
    logger.info('End of evaluation...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
