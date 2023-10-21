import torch
from torch.utils.data import Dataset

import pandas as pd
import sys
import os
from time import time

import problems.utils_data as utils
import problems.utils_edges_street as get_edges_street
import problems.utils_edges_euclidean as get_edges_euclidean
from utils.config import *

class DSVRPTW_Dataset(Dataset):
    customer_feature = 7  # customer features location (x_i,y_i) and duration of service(d), appearance (u)

    @classmethod
    def create_data(cls,
                    batch_size=1,
                    customer_count=100,
                    vehicle_count=25,
                    vehicle_capacity=200,
                    vehicle_speed=1,
                    customer_location_range=(0, 101),
                    customer_demand_range=(5, 41),
                    horizon=480,
                    customer_duration_range=(10, 31),
                    tw_ratio=0.5,
                    customer_tw_range=(30, 91),
                    dod=0.5,
                    d_early_ratio=0.5
                    ):

        size = (batch_size, customer_count, 1)

        # Sample locs        x_j, y_j ~ U(0, 100)
        locations = torch.randint(*customer_location_range, (batch_size, customer_count + 1, 2), dtype=torch.float)
        # Sample dems             q_j ~ U(5,  40)
        demands = torch.randint(*customer_demand_range, size, dtype=torch.float)
        # Sample serv. time       s_j ~ U(10, 30)
        durations = torch.randint(*customer_duration_range, size, dtype=torch.float)

        # Sample dyn subset           ~ B(dod)
        # and early/late appearance   ~ B(d_early_ratio)
        if isinstance(dod, float):
            is_dyn = torch.empty(size).bernoulli_(dod)
        elif len(dod) == 1:
            is_dyn = torch.empty(size).bernoulli_(dod[0])
        else:  # tuple of float
            ratio = torch.tensor(dod)[torch.randint(0, len(dod), (batch_size,), dtype=torch.int64)]
            is_dyn = ratio[:, None, None].expand(*size).bernoulli()


        if isinstance(d_early_ratio, float):
            is_dyn_e = torch.empty(size).bernoulli_(d_early_ratio)
        elif len(d_early_ratio) == 1:
            is_dyn_e = torch.empty(size).bernoulli_(d_early_ratio[0])
        else:
            ratio = torch.tensor(d_early_ratio)[torch.randint(0, len(d_early_ratio), (batch_size,), dtype=torch.int64)]
            is_dyn_e = ratio[:, None, None].expand(*size).bernoulli()

        # Sample appear. time     a_j = 0 if not in D subset
        #                         a_j ~ U(1,H/3) if early appear
        #                         a_j ~ U(H/3+1, 2H/3) if late appear
        appears = is_dyn * is_dyn_e * torch.randint(1, horizon // 3 + 1, size, dtype=torch.float) \
                  + is_dyn * (1 - is_dyn_e) * torch.randint(horizon // 3 + 1, 2 * horizon // 3 + 1, size, dtype=torch.float)

        # Sample TW subset            ~ B(tw_ratio)
        if isinstance(tw_ratio, float):
            has_tw = torch.empty(size).bernoulli_(tw_ratio)
        elif len(tw_ratio) == 1:
            has_tw = torch.empty(size).bernoulli_(tw_ratio[0])
        else:  # tuple of float
            ratio = torch.tensor(tw_ratio)[torch.randint(0, len(tw_ratio), (batch_size,), dtype=torch.int64)]
            has_tw = ratio[:, None, None].expand(*size).bernoulli()

        # Sample TW width        tw_j = H if not in TW subset
        #                        tw_j ~ U(30,90) if in TW subset
        tws = (1 - has_tw) * torch.full(size, horizon) \
              + has_tw * torch.randint(*customer_tw_range, size, dtype=torch.float)

        tts = (locations[:, None, 0:1, :] - locations[:, 1:, None, :]).pow(2).sum(-1).pow(0.5) / vehicle_speed
        # Sample ready time       e_j = 0 if not in TW subset
        #                         e_j ~ U(a_j, H - max(tt_0j + s_j, tw_j))
        rdys = has_tw * (appears + torch.rand(size) * (horizon - torch.max(tts + durations, tws) - appears))
        rdys.floor_()

        # Regroup all features in one tensor
        customers = torch.cat((locations[:, 1:], demands, rdys, rdys + tws, durations, appears), 2)

        # Add depot node
        depot_node = torch.zeros((batch_size, 1, cls.customer_feature))
        depot_node[:, :, :2] = locations[:, 0:1]
        depot_node[:, :, 4] = horizon
        nodes = torch.cat((depot_node, customers), 1)
        customer_mask = None

        dataset = cls(vehicle_count, vehicle_capacity, vehicle_speed, nodes, customer_mask)

        return dataset

    def __init__(self, vehicle_count, vehicle_capacity, vehicle_speed, nodes, customer_mask=None):

        self.vehicle_count = vehicle_count
        self.vehicle_capacity = vehicle_capacity
        self.vehicle_speed = vehicle_speed
        self.nodes = nodes

        self.batch_size, self.nodes_count, d = self.nodes.size()

        if d != self.customer_feature:
            raise ValueError("Expected {} customer features per nodes, got {}".format(
                self.customer_feature, d))

        self.customer_mask = customer_mask

    def __len__(self):
        return self.batch_size

    def __getitem__(self, i):
        if self.customer_mask is None:
            return self.nodes[i]
        else:
            return self.nodes[i], self.customer_mask[i]

    def nodes_generate(self):
        if self.customer_mask is None:
            yield from self.nodes
        else:
            yield from (n[m ^ 1] for n, m in zip(self.nodes, self.customer_mask))

    def normalize(self):
        loc_scl, loc_off = self.nodes[:, :, :2].max().item(), self.nodes[:, :, :2].min().item()
        loc_scl -= loc_off
        t_scl = self.nodes[:, 0, 4].max().item()

        self.nodes[:, :, :2] -= loc_off
        self.nodes[:, :, :2] /= loc_scl
        self.nodes[:, :, 2] /= self.vehicle_capacity
        self.nodes[:, :, 3:] /= t_scl

        self.veh_capa = 1
        self.vehicle_speed *= t_scl / loc_scl
        return loc_scl, t_scl

    def save(self, folder_path):
        torch.save({
            'vehicle_count': self.vehicle_count,
            'vehicle_capacity':self.vehicle_capacity,
            'vehicle_speed': self.vehicle_speed,
            'nodes': self.nodes,
            'cust_mask': self.customer_mask
        }, folder_path)

    @classmethod
    def load(cls, folder_path):
        return cls(**torch.load(folder_path))

if __name__ == '__main__':
    args = ParseArguments()
    start_time = time()

    train_test_val = 'validation'

    if train_test_val == 'train':
        batch_size = args.batch_size * args.iter_count
    elif train_test_val == 'test':
        batch_size = args.test_batch_size
    else:
        batch_size = 10

    print(batch_size)
    vehicle_count = args.vehicle_count
    vehicle_capacity = args.vehicle_capacity
    vehicle_speed = args.vehicle_speed
    customer_count = args.customer_count
    dod = args.dod
    horizon = args.horizon
    data = DSVRPTW_Dataset.create_data(batch_size=batch_size,
                                       customer_count=customer_count,
                                       vehicle_count=vehicle_count,
                                       dod=dod)
    if train_test_val == 'train':
        data.normalize()

    end_time = time()

    # save the data
    folder_path = "../data/{}/{}".format(train_test_val, customer_count)
    os.makedirs(folder_path, exist_ok=True)
    if train_test_val == 'train':
        torch.save(data, os.path.join(folder_path, "train.pth"))
    elif train_test_val == 'test':
        torch.save(data, os.path.join(folder_path, "test.pth"))
    else:
        torch.save(data, os.path.join(folder_path, "val.pth"))

    print(f'Time to run {batch_size} batches is {end_time - start_time}')
    print(data.nodes[0], data.nodes[0].size(), data.vehicle_speed)
