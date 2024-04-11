from collections import Counter

import torch
from torch.utils.data import Dataset

from utils import ReadFile


class InteractionData(Dataset):
    def __init__(self, args):
        self.args = args
        train_df, test_df = ReadFile.read_interaction_csv(args.train_path), ReadFile.read_interaction_csv(args.test_path)
        self.num_user, self.num_item = train_df['u'].nunique(), train_df['i'].nunique()
        if args.dataset == 'ml-1m':
            self.num_user, self.num_item = 5148, 2380
        self.user_list, self.item_list = train_df['u'].values, train_df['i'].values
        print('=' * 10)
        print(f'# User: {self.num_user}\n# Item: {self.num_item}')
        print(f'# Train: {len(train_df)}\n# Test: {len(test_df)}')
        print('-' * 10)

        self.record = dict()
        for u, i in zip(train_df['u'], train_df['i']):
            if self.record.get(u) is None:
                self.record[u] = [i]
            else:
                self.record[u].append(i)
        
        self.ground_truth = dict()
        if args.dataset == 'ml-1m':
            item_set_of_train = set(train_df['i'].values)
            for u, i in zip(test_df['u'], test_df['i']):
                if i in item_set_of_train:
                    if self.ground_truth.get(u) is None:
                        self.ground_truth[u] = [i]
                    else:
                        self.ground_truth[u].append(i)
        else:
            for u, i in zip(test_df['u'], test_df['i']):
                if self.ground_truth.get(u) is None:
                    self.ground_truth[u] = [i]
                else:
                    self.ground_truth[u].append(i)
        
        self.record_set = dict()
        for u, i in zip(train_df['u'], train_df['i']):
            if self.record_set.get(u) is None:
                self.record_set[u] = {i}
            else:
                self.record_set[u].add(i)

        self.record_item_set = dict()
        for u, i in zip(train_df['u'], train_df['i']):
            if self.record_item_set.get(i) is None:
                self.record_item_set[i] = {u}
            else:
                self.record_item_set[i].add(u)
        
        interaction_dict = {(user, item): 1 for user, item in zip(self.user_list, self.item_list)}
        self.users, self.items, self.labels = [], [], []
        for user in range(self.num_user):
            for item in range(self.num_item):
                self.users.append(user)
                self.items.append(item)
                self.labels.append(interaction_dict.get((user, item), 0))

        item_counter = Counter(self.item_list)
        self.item_pop = torch.tensor([item_counter[i] for i in range(self.num_item)]).to(args.device)
        self.theta = (self.item_pop / self.item_pop.max()).pow(args.beta).to(args.device)
        self.theta = torch.clip(self.theta, 0.1, 1.0)
        self.theta_n = 1 - self.theta
        self.theta_n = torch.clip(self.theta_n, 0.1, 1.0)
        self.max_item_pop = self.item_pop.max()

    def __len__(self):
        if self.args.type == 'pair':
            return len(self.user_list)
        else:
            return self.num_user * self.num_item

    def __getitem__(self, idx):
        if self.args.type == 'pair':
            uid = self.user_list[idx]
            iid = self.item_list[idx]
            return dict(uid=uid, iid=iid)
        else:
            uid = self.users[idx]
            iid = self.items[idx]
            y = self.labels[idx]
            return dict(uid=uid, iid=iid, y=y)
