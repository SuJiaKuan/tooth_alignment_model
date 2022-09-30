import argparse
import os
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
from utils.utils import test, save_checkpoint
from model.pointconv import PointConvDensityClsSsg as PointConvClsSsg
import provider
import numpy as np 


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointConv')
    parser.add_argument('data_path', type=str, help='path to dataset')
    parser.add_argument('--values', type=int, nargs='+', default=[0, 2, 3, 4, 5, 6], choices=list(range(7)), help='target values')
    parser.add_argument('--metric', type=str, default='r2_score', choices=['mse', 'mae', 'r2_score'], help='target testing metric to decide the best checkpoint')
    parser.add_argument('--batchsize', type=int, default=32, help='batch size in training')
    parser.add_argument('--epoch',  default=400, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--num_workers', type=int, default=16, help='Worker Number [default: 16]')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'], help='optimizer for training')
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate of learning rate')
    parser.add_argument('--model_name', default='pointconv', help='model name')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    return parser.parse_args()


def weighted_mse_loss(pred, target, weights):
    return (weights * (pred - target) ** 2).mean()


def log_cosh_loss(pred, target):
    ey_t = pred - target

    return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%s_ModelNet40-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'train_%s_cls.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())
    logger.info('---------------------------------------------------TRANING---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    '''DATA LOADING'''
    logger.info('Load dataset ...')

    train_dataset = ModelNetDataLoader(args.data_path, args.values, npoint=args.num_point, split='train', normal_channel=args.normal)
    test_dataset = ModelNetDataLoader(args.data_path, args.values, npoint=args.num_point, split='test', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers)

    logger.info("The number of training data is: %d", len(train_dataset))
    logger.info("The number of test data is: %d", len(test_dataset))

    seed = 3
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    '''MODEL LOADING'''
    classifier = PointConvClsSsg(len(args.values)).cuda()
    if args.pretrain is not None:
        logger.info('Use pretrain model')
        checkpoint = torch.load(args.pretrain)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0


    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_test_score = float('-inf') if args.metric == 'r2_score' else float('inf')

    weights = torch.Tensor([5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0]).cuda()

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        logger.info('Epoch %d (%d/%s):' ,global_epoch + 1, epoch + 1, args.epoch)

        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            tooth_pcs, jaw_pc, target = data
            tooth_pcs = tooth_pcs.transpose(3, 2)
            jaw_pc = jaw_pc.transpose(2, 1)
            tooth_pcs, jaw_pc, target = tooth_pcs.cuda(), jaw_pc.cuda(), target.cuda()
            optimizer.zero_grad()

            classifier = classifier.train()
            pred = classifier(tooth_pcs, jaw_pc)
            loss = log_cosh_loss(pred, target)
            loss.backward()
            optimizer.step()
            global_step += 1

        test_metric = test(classifier, testDataLoader)

        test_score = test_metric[args.metric]
        is_better = \
            test_score >= best_test_score \
            if args.metric == 'r2_score' \
            else test_score <= best_test_score

        if is_better and epoch > 5:
            best_test_score = test_score
            logger.info('Save model...')
            save_checkpoint(
                global_epoch + 1,
                loss,
                test_score,
                test_metric,
                classifier,
                optimizer,
                str(checkpoints_dir),
                args.model_name,
            )

        logger.info('Loss: %.2f', loss.data)
        logger.info('Test Score: %f  *** Best Test Score: %f', test_score, best_test_score)
        logger.info('Evaluation Metrics:')
        logger.info(test_metric)

        global_epoch += 1
    logger.info('Best Score: %f', best_test_score)

    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)
