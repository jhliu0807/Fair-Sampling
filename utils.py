import os
import random
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    @staticmethod
    def forward(pos_scores, neg_scores, weight=None):
        if weight is None:
            return torch.sum(torch.nn.functional.softplus(neg_scores - pos_scores))
        else:
            return torch.sum(torch.nn.functional.softplus(neg_scores - pos_scores) * weight)


class ModelAccess:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def save_checkpoint(model, folder, filename):
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(model, f'{folder}/{filename}')

    @staticmethod
    def load_checkpoint(filepath):
        return torch.load(filepath, map_location=torch.device('cpu'))


class ReadFile:
    def __init__(self) -> None:
        pass

    @staticmethod
    def read_interaction_csv(filepath, sep=',') -> pd.DataFrame:
        return pd.read_csv(filepath, names=['u', 'i'], sep=sep)


def get_prefix(args):
    # dataset, model(dim,beta)
    dataset = f'{args.dataset}'
    model = f'{args.model}-{args.dim}-{args.beta}'
    prefix = f'{dataset}-{model}'
    return prefix


def build_log_path(args):
    now = datetime.now()
    time_str = now.strftime("%Y%m%d-%H%M%S")
    prefix = get_prefix(args)
    return f'{args.logdir}/{prefix}_{time_str}'


def evaluate(dataset, model, args):
    Recall, NDCG, Precise, F1, ARP = 0., 0., 0., 0., 0.
    pbar = tqdm(enumerate(dataset.ground_truth.items()), total=len(dataset.ground_truth))
    for it, item in pbar:
        u, gt = item
        score_pred = model.get_single_user_score(u).cpu()
        score_pred[dataset.record[u]] = -torch.inf
        _, pred_idx_single = score_pred.topk(args.topk)
        pred_idx_single = pred_idx_single.tolist()
        arp = dataset.item_pop[pred_idx_single].sum()
        ARP += arp
        DCG, IDCG = 0., 0.
        for i in range(args.topk):
            if pred_idx_single[i] in gt:
                DCG += (1 / np.log(2 + i))
        for i in range(min(args.topk, len(gt))):
            IDCG += (1 / np.log(2 + i))
        NDCG += (DCG / IDCG)
        right = len(set(pred_idx_single) & set(gt))
        recall = (right / len(gt))
        precice = (right / args.topk)
        Recall += recall
        Precise += precice
        if recall + precice != 0:
            F1 += (2 * recall * precice) / (recall + precice)
        pbar.set_description(f'[Test ] iter {it}')
    Recall /= len(dataset.ground_truth)
    NDCG /= len(dataset.ground_truth)
    Precise /= len(dataset.ground_truth)
    F1 /= len(dataset.ground_truth)
    ARP /= len(dataset.ground_truth)
    print(f'{Recall:.4f};{NDCG:.4f};{Precise:.4f};{F1:.4f};{ARP:.4f}')
    return Recall, NDCG, Precise, F1, ARP
