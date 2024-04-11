import argparse
from pprint import pprint

import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='yelp', type=str, choices=['yelp', 'kindle', 'gowalla', 'ml-1m'])
    parser.add_argument('--model', default='BPR', choices=['BPR', 'UBPR', 'CPR', 'DICE', 'Pair', 'WMF', 'Rel-MF', 'DU', 'Point'], help='training strategy')
    parser.add_argument('--beta', default=0.5, type=float, help='hyperparameter for IPS-based method')
    parser.add_argument('--phase', default='train', type=str, choices=['train', 'test'])
    parser.add_argument('--cuda', default=0, type=int, help='cuda index, <0 means use cpu')
    parser.add_argument('--dim', default=64, type=int, help='embedding dimension')
    parser.add_argument('--decay', default=1e-3, type=float, help='the weight decay for l2 normalization')
    parser.add_argument('--topk', default=20, type=int, help='top k')
    parser.add_argument('--batch', default=1024, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
    parser.add_argument('--epoch', default=1000, type=int, help='the number of epoch')
    parser.add_argument('--patience', default=20, type=int, help='early stop')
    parser.add_argument('--test', default=1, type=int, help='epoch per test')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--logdir', default='./runs', type=str, help='the log direction for tensorboard')
    parser.add_argument('--ckpt', default='./ckpt', type=str, help='checkpoint folder')
    args = parser.parse_args()
    args.train_path = f'../input/processed/{args.dataset}/train.csv'
    args.test_path = f'../input/processed/{args.dataset}/valid.csv' if args.phase == 'train' else  f'../input/processed/{args.dataset}/test.csv'
    args.device = torch.device('cuda:{}'.format(args.cuda)) if args.cuda >= 0 else torch.device('cpu')
    args.type = 'point' if args.model in ['WMF', 'Rel-MF', 'DU', 'Point'] else 'pair'
    print('=' * 10)
    pprint(args.__dict__)
    print('-' * 10)
    return args
