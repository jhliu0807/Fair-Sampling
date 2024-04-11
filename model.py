import torch
import torch.nn as nn


class MF(nn.Module):
    def __init__(self, num_users, num_items, dim):
        super(MF, self).__init__()
        self.user_embedding = nn.Parameter(torch.normal(mean=0., std=1, size=(num_users, dim)))
        self.item_embedding = nn.Parameter(torch.normal(mean=0., std=1, size=(num_items, dim)))

    def forward(self, uid_batch, iid_batch):
        u_batch = self.user_embedding[uid_batch]
        i_batch = self.item_embedding[iid_batch]
        return torch.sum(u_batch * i_batch, dim=1)

    def get_single_user_score(self, uid):
        return self.user_embedding[uid] @ self.item_embedding.T


class DICE(nn.Module):
    def __init__(self, num_users, num_items, dim):
        super(DICE, self).__init__()
        self.user_int = nn.Parameter(torch.normal(mean=0., std=1, size=(num_users, dim//2)))
        self.item_int = nn.Parameter(torch.normal(mean=0., std=1, size=(num_items, dim//2)))
        self.user_pop = nn.Parameter(torch.normal(mean=0., std=1, size=(num_users, dim//2)))
        self.item_pop = nn.Parameter(torch.normal(mean=0., std=1, size=(num_items, dim//2)))

    def cal_int(self, uid_batch, iid_batch):
        u_batch_int = self.user_int[uid_batch]
        i_batch_int = self.item_int[iid_batch]
        return torch.sum(u_batch_int * i_batch_int, dim=1)
    
    def cal_pop(self, uid_batch, iid_batch):
        u_batch_pop = self.user_pop[uid_batch]
        i_batch_pop = self.item_pop[iid_batch]
        return torch.sum(u_batch_pop * i_batch_pop, dim=1)

    def get_single_user_score(self, uid):
        return self.user_int[uid] @ self.item_int.T
