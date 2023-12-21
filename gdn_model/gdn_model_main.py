import torch.nn.functional as F
import torch
from models.GDN import GDN
from util.data import *
import pandas as pd
from torch.utils.data import DataLoader, Subset
from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc
from datasets.TimeDataset import TimeDataset
from models.GDN import GDN


class GDNMain():
    def __init__(self, train_config, env_config, debug=False):

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None
        dataset = self.env_config['dataset'] 
        train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
        test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0)
        train, test = train_orig, test_orig
        self.test = test
        self.modified_test = test.copy()

        if 'attack' in train.columns:
            train = train.drop(columns=['attack'])

        feature_map = get_feature_map(dataset)
        self.fc_struc = get_fc_graph_struc(dataset)

        set_device(env_config['device'])
        self.device = get_device()

        self.train = train.copy()

        self.fc_edge_index = build_loc_net(self.fc_struc, list(self.train.columns), feature_map=feature_map)
        self.fc_edge_index = torch.tensor(self.fc_edge_index, dtype = torch.long)

        self.feature_map = feature_map

        train_dataset_indata = construct_data(train, feature_map, labels=0)
        test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())

        self.cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }

        train_dataset = TimeDataset(train_dataset_indata, self.fc_edge_index, mode='train', config=self.cfg)
        test_dataset = TimeDataset(test_dataset_indata, self.fc_edge_index, mode='test', config=self.cfg)

        train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'], val_ratio = train_config['val_ratio'])

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'], shuffle=False, num_workers=0)


        edge_index_sets = []
        edge_index_sets.append(self.fc_edge_index)
        self.edge_index_sets = edge_index_sets.copy()

        self.model = GDN(edge_index_sets, len(feature_map), 
                dim=train_config['dim'], 
                input_dim=train_config['slide_win'],
                out_layer_num=train_config['out_layer_num'],
                out_layer_inter_dim=train_config['out_layer_inter_dim'],
                topk=train_config['topk']
            ).to(self.device)
        
    def initialize_GDN(self):
        return self.model

    def get_test_dataset(self):
        return self.test_dataloader
    
    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        # val_start_index = random.randrange(train_use_len)
        val_start_index = 3890
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)


        train_dataloader = DataLoader(train_subset, batch_size=batch,
                                shuffle=True)

        val_dataloader = DataLoader(val_subset, batch_size=batch,
                                shuffle=False)

        return train_dataloader, val_dataloader
    
    def get_dataloader(self, dataset):
        '''After modifying the data in the dataset, convert the dataset back to dataloader for model to make prediction'''

        new_dataset = construct_data(dataset, self.feature_map, labels=dataset.attack.tolist())
        new_dataset = TimeDataset(new_dataset, self.fc_edge_index, mode='test', config=self.cfg)
        new_dataset = DataLoader(new_dataset, batch_size=self.train_config['batch'], shuffle=False, num_workers=0)
        
        return new_dataset
    
    def get_test_data(self):
        '''Test data without attack payload added.'''
        return self.test.copy()