import argparse
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
from tqdm import tqdm
from model.pointconv import PointConvDensityClsSsg as PointConvClsSsg


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointConv')
    parser.add_argument('data_path', type=str, help='path to dataset')
    parser.add_argument('--values', type=int, nargs='+', default=[0, 2, 3, 4, 5, 6], choices=list(range(7)), help='target values')
    parser.add_argument('--batchsize', type=int, default=32, help='batch size')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--num_workers', type=int, default=16, help='Worker Number [default: 16]')
    parser.add_argument('--model_name', default='pointconv', help='model name')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    return parser.parse_args()


def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = Path('./eval_experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%s_ModelNet40-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    os.system('cp %s %s' % (args.checkpoint, checkpoints_dir))
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'eval_%s_cls.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
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
    classifier = PointConvClsSsg(values).cuda()
    if args.checkpoint is not None:
        print('Load CheckPoint...')
        logger.info('Load CheckPoint')
        checkpoint = torch.load(args.checkpoint)
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('Please load Checkpoint to eval...')
        sys.exit(0)

    '''EVAL'''
    logger.info('Start evaluating...')
    print('Start evaluating...')

    classifier = classifier.eval()
    mses = []
    maes = []
    sub_maes = []
    for batch_id, data in tqdm(enumerate(testDataLoader, 0), total=len(testDataLoader), smoothing=0.9):
        tooth_pcs, jaw_pc, target = data
        tooth_pcs = tooth_pcs.transpose(3, 2)
        jaw_pc = jaw_pc.transpose(2, 1)
        tooth_pcs, jaw_pc, target = tooth_pcs.cuda(), jaw_pc.cuda(), target.cuda()

        with torch.no_grad():
            pred = classifier(tooth_pcs, jaw_pc)

        mse = F.mse_loss(pred, target)
        mses.append(mse.item())

        mae = F.l1_loss(pred, target)
        maes.append(mae.item())

        sub_mae = \
            np.mean(np.abs(pred.cpu().numpy() - target.cpu().numpy()), axis=0)
        sub_maes.append(sub_mae)

    test_mse = np.mean(mses)
    test_mae = np.mean(maes)
    test_sub_mae = np.mean(sub_maes, axis=0)

    print('Test MSE {}'.format(test_mse))
    print('Test MAE {}'.format(test_mae))
    print('Test SUB-MAE {}'.format(test_sub_mae))

    logger.info('Test MSE {}'.format(test_mse))
    logger.info('Test MAE {}'.format(test_mae))
    logger.info('Test SUB-MAE {}'.format(test_sub_mae))
    logger.info('End of evaluation...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
