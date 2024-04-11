import time

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils import ModelAccess, get_prefix, build_log_path, evaluate
from c_utils import sampler


class Processor:
    def __init__(self, model, criterion, optimizer, dataloader, dataset, args):
        self.model, self.criterion, self.optimizer = model, criterion, optimizer
        self.dataloader, self.dataset = dataloader, dataset
        self.args = args
        self.writer = SummaryWriter(build_log_path(args))
        self.patience = self.args.patience
        self.best_recall, self.best_ndcg, self.best_precise, self.best_f1 = 0, 0, 0, 0
        self.my_sampler = sampler.InteractionSampler(dataset.record_set, dataset.record_item_set, dataset.num_user, dataset.num_item)
    
    def process(self):
        self.test(0)
        for epoch in range(self.args.epoch):
            if self.args.type == 'pair':
                self.train_one_epoch(epoch+1)
            else:
                self.train_one_epoch_point(epoch+1)
            if (epoch+1) % self.args.test == 0:
                save, stop = self.test(epoch+1)
                if save:
                    ModelAccess.save_checkpoint(self.model, self.args.ckpt, get_prefix(self.args)+'.pth')
                if stop:
                    break
        time.sleep(1)
        print('Done.')

    def train_one_epoch(self, epoch):
        total_loss = 0
        pbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
        decay = np.power(0.99, epoch - 1)  # for DICE
        for it, data_batch in pbar:
            uid_batch, pos_iid_batch = data_batch['uid'].tolist(), data_batch['iid'].tolist()

            if self.args.model == 'BPR':
                neg_list = self.my_sampler.negative_sample(uid_batch)
                pos_score = self.model(uid_batch, pos_iid_batch)
                neg_score = self.model(uid_batch, neg_list)
                loss = self.criterion(pos_score, neg_score)
            elif self.args.model == 'UBPR':
                iid_list, label_list = self.my_sampler.sample(uid_batch)
                label_list = torch.tensor(label_list, dtype=torch.float32).to(self.args.device)
                pos_score = self.model(uid_batch, pos_iid_batch)
                score = self.model(uid_batch, iid_list)
                weight = 1 / self.dataset.theta[pos_iid_batch] * (1 - label_list / self.dataset.theta[iid_list])
                loss = self.criterion(pos_score, score, weight)
            elif self.args.model == 'DICE':
                neg_list = self.my_sampler.negative_sample(uid_batch)
                pos_int_score = self.model.cal_int(uid_batch, pos_iid_batch)
                pos_pop_score = self.model.cal_pop(uid_batch, pos_iid_batch)
                neg_int_score = self.model.cal_int(uid_batch, neg_list)
                neg_pop_score = self.model.cal_pop(uid_batch, neg_list)
                pos_score = pos_int_score + pos_pop_score
                neg_score = neg_int_score + neg_pop_score
                loss_click = self.criterion(pos_score, neg_score)
                thredhold = 0.1 * decay * self.dataset.max_item_pop
                diff = self.dataset.item_pop[pos_iid_batch] - self.dataset.item_pop[neg_list]
                weight_int = (diff < -thredhold).float()
                loss_int = self.criterion(pos_int_score, neg_int_score, weight_int)
                weight_pop = torch.zeros_like(diff)
                weight_pop[diff > thredhold] = 1
                weight_pop[diff < -thredhold] = -1
                loss_pop = self.criterion(pos_pop_score, neg_pop_score, weight_pop)
                loss = loss_click + self.args.beta * decay * (loss_int + loss_pop)
            elif self.args.model == 'CPR':
                u2_list, i2_list = self.my_sampler.fair_sample(uid_batch, pos_iid_batch)
                p_score1 = self.model(uid_batch, pos_iid_batch)
                n_score1 = self.model(uid_batch, i2_list)
                p_score2 = self.model(u2_list, i2_list)
                n_score2 = self.model(u2_list, pos_iid_batch)
                loss = self.criterion(p_score1+p_score2, n_score1+n_score2)
            elif self.args.model == 'Pair':
                u2_list, i2_list = self.my_sampler.fair_sample(uid_batch, pos_iid_batch)
                p_score1 = self.model(uid_batch, pos_iid_batch)
                n_score1 = self.model(uid_batch, i2_list)
                p_score2 = self.model(u2_list, i2_list)
                n_score2 = self.model(u2_list, pos_iid_batch)
                loss1 = self.criterion(p_score1, n_score1)
                loss2 = self.criterion(p_score2, n_score2)
                loss = loss1 + loss2
            
            total_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pbar.set_description(f'[Train] epoch {epoch} it {it} loss {loss.item():.4f}')
        total_loss /= len(self.dataloader)
        self.writer.add_scalar('Loss/Train', total_loss, epoch)

    def train_one_epoch_point(self, epoch):
        total_loss = 0
        pbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
        for it, batch in pbar:
            user_batch, item_batch, Y = batch['uid'].tolist(), batch['iid'].tolist(), batch['y'].to(self.args.device)
            Y_hat = self.model(user_batch, item_batch)
            Y_hat = nn.Sigmoid()(Y_hat)
            if self.args.model == 'WMF':
                loss =  ((Y_hat - Y)**2).sum()
            elif self.args.model == 'Rel-MF':
                loss =  ((1 - Y) * Y_hat**2).sum()
                loss += (Y * (1 / self.dataset.theta[item_batch] * (1 - Y_hat)**2 + (1 - 1 / self.dataset.theta[item_batch]) * Y_hat**2)).sum()
            elif self.args.model == 'DU':
                loss =  ((1 - Y) * 1 / self.dataset.theta_n[item_batch] * Y_hat**2).sum()
                loss += (Y * 1 / self.dataset.theta[item_batch] * (1 - Y_hat)**2).sum()
            elif self.args.model == 'Point':
                loss =  ((Y_hat - Y)**2).sum()
                u2, i2, Y2 = self.my_sampler.fair_sample_point(user_batch, item_batch, Y.tolist())
                Y2 = torch.tensor(Y2).to(self.args.device)
                Y2_hat = self.model(u2, i2)
                Y2_hat = nn.Sigmoid()(Y2_hat)
                loss += ((Y2_hat - Y2)**2).sum()
            total_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pbar.set_description(f'[Train] epoch {epoch} it {it} loss {loss.item():.4f}')
        total_loss /= len(self.dataloader)
        self.writer.add_scalar('Loss/Train', total_loss, epoch)

    def test(self, epoch):
        Recall, NDCG, Precise, F1, ARP = evaluate(self.dataset, self.model, self.args)
        
        self.writer.add_scalar(f'Metric/Recall', Recall, epoch)
        self.writer.add_scalar(f'Metric/NDCG', NDCG, epoch)
        self.writer.add_scalar(f'Metric/ARP', ARP, epoch)
        
        if Recall > self.best_recall:
            self.best_recall, self.best_ndcg, self.best_precise, self.best_f1 = Recall, NDCG, Precise, F1
            self.patience = self.args.patience
            return True, False
        else:
            self.patience -= 1
            if self.patience == 0:
                return False, True
            else:
                return False, False
