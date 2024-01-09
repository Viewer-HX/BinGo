import os
import numpy as np
import torch
from random import shuffle
from torch_geometric.data import Data
from natsort import natsorted

class PairData(Data):
    def __init__(self, edge_index_s, edge_attr_s, x_s, edge_index_t, edge_attr_t, x_t, y):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.edge_attr_s = edge_attr_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.edge_attr_t = edge_attr_t
        self.x_t = x_t
        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super(PairData, self).__inc__(key, value)

def GetDataset(path=None):
    '''
    Get the dataset from numpy data files.
    :param path: the path used to store numpy dataset.
    :return: dataset - list of torch_geometric.data.Data
    '''

    # check.
    if None == path:
        print('[Error] <GetDataset> The method is missing an argument \'path\'!')
        return [], []

    # get ordered file list.
    filelist = []
    for root, _, fs in os.walk(path):
        for file in fs:
            if '.DS_Store' in file: continue
            filename = os.path.join(root, file).replace('\\', '/')
            filelist.append(filename)
    filelist = natsorted(filelist)
    shuffle(filelist)

    # contruct the dataset.
    dataset = []
    for filename in filelist:
        # print(filename)
        # read a numpy graph file.
        graph = np.load(filename, allow_pickle=True)
        # sparse each element.
        edgeIndex0 = torch.tensor(graph['edgeIndex0'], dtype=torch.long)
        edgeIndex1 = torch.tensor(graph['edgeIndex1'], dtype=torch.long)
        edgeAttr0 = torch.tensor(graph['edgeAttr0'], dtype=torch.float)
        edgeAttr1 = torch.tensor(graph['edgeAttr1'], dtype=torch.float)
        nodeAttr0 = torch.tensor(graph['nodeAttr0'], dtype=torch.float)
        nodeAttr1 = torch.tensor(graph['nodeAttr1'], dtype=torch.float)
        label = torch.tensor(graph['label'], dtype=torch.long)
        # construct an instance of torch_geometric.data.Data.
        data = PairData(edge_index_s=edgeIndex0, edge_attr_s=edgeAttr0, x_s=nodeAttr0,
                        edge_index_t=edgeIndex1, edge_attr_t=edgeAttr1, x_t=nodeAttr1, y=label)
        # append the Data instance to dataset.
        dataset.append(data)

    if (0 == len(dataset)):
        print(f'[ERROR] Fail to load data from {path}')

    return dataset, filelist