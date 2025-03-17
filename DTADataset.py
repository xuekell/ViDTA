import dgl
import pandas as pd
import torch
import numpy as np
import pickle
import h5py

from dgl import load_graphs
from torch.utils.data import DataLoader, Dataset

if torch.cuda.is_available():
    device = torch.device('cuda')

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25


def seq2int(line):

    return [CHARPROTSET[s] for s in line.upper()]

class DTADataset(Dataset):
    def __init__(self, dataset=None, dataset_fold=None):
        self.dataset = dataset
        self.data = dataset_fold
        self.compound_graph = self.compound_graph_get()
        self.target = self.target_get()
        self.label = self.data[:, -1]


    def __len__(self):
        return len(self.label)


    def __getitem__(self, idx):

        compound_len = self.compound_graph[idx].num_nodes()

        return self.compound_graph[idx], self.target[idx], compound_len, self.label[idx]

    ''' 整数编码 '''
    def target_get(self):
        
        N = len(self.data)
        target_list = []
        sequence = self.data[:, 1]
        target_len = 1000
        for protein in sequence:
            target = seq2int(protein)
            if len(target) < target_len:
                target = np.pad(target, (0, target_len - len(target)))
            else:
                target = target[:target_len]
            target_list.append(target)        
        return target_list
    
    def compound_graph_get(self):
        smiles_TVdataset = self.data[:, 0]
        compounds_graph_TVdataset = []
        # N = len(id_TVdataset)
        with open('dataset/'+ self.dataset + '/processed/compound_graphs_vn.pkl', 'rb') as f:  # 使用 pickle 从文件中加载字典对象
            smiles2graph = pickle.load(f)
        for no, smile in enumerate(smiles_TVdataset):
            # print('/'.join(map(str, [no + 1, N])))
            # compound_graph_TVdataset, _ = load_graphs('dataset/' + self.dataset + '/processed' + '/compound_graph/' + str(id) + '.bin')
            compound_graph_TVdataset = smiles2graph[smile]
            compounds_graph_TVdataset.append(compound_graph_TVdataset[0])
        return compounds_graph_TVdataset


    def collate(self, sample):
        # batch_size = len(sample)

        compound_graph, target, compound_len, label = map(list, zip(*sample))
        
        compound_graph = dgl.batch(compound_graph)
        label = torch.FloatTensor(label)
        return compound_graph, target, label



