import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import torch_geometric.nn as gnn
import torch_geometric
from config import config
import timm
import networkx as nx



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NUN(nn.Module):
    def __init__(self, in_channels, num_classes, out_size=9):
        super(NUN, self).__init__()
        self.backbone = timm.create_model('inception_v3', pretrained=True)
        self.backbone.fc = Identity()
        self.backbone.global_pool = Identity()
        self.edges = None
        nodes = out_size**2
        complete_graph_edges = int((nodes * (nodes - 1)) / 2)

        self.generate_graph(nodes)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_weights = nn.Linear(2048, complete_graph_edges)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm1d(complete_graph_edges)

        self.conv1 = gnn.GraphConv(2048, 1024, aggr='add')
        self.norm1 = gnn.BatchNorm(1024)
        self.conv2 = gnn.GraphConv(1024, 1024, aggr='add')
        self.norm2 = gnn.BatchNorm(1024)
        self.fc = nn.Linear(1024, num_classes)


    def generate_graph(self, nodes):
        g = nx.complete_graph(nodes)
        self.edges = list(g.edges)
        self.edges = torch.LongTensor(self.edges)

        return

    def create_graph(self, x):
        arr = []
        for i, weights in zip(x, self.edge_arr):
            data = Data(x=i, edge_index=self.edges.t().contiguous(), edge_weight=weights.t().contiguous())
            arr.append(data)
        b = Batch.from_data_list(arr)
        return b

    def g_forward(self, data):
        x = data.x

        edge_index = data.edge_index
        edge_weight = data.edge_weight
        batch = data.batch
        x = self.conv1(x, edge_index, edge_weight)
        x = self.norm1(x)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight)
        x = self.norm2(x)
        x = x.relu()

        x = gnn.global_mean_pool(x, batch)

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc(x)

        return x


    def forward(self, x):
        self.edge_arr = []
        x = self.backbone(x)
        print(x.shape)
        self.create_edge_weights(x)

        x = torch.flatten(x, start_dim=2)
        x = torch.swapaxes(x, 1, -1)

        data = self.create_graph(x)

        data = data.cuda()
        out = self.g_forward(data)

        return out

    def create_edge_weights(self, x):

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.conv_weights(x)
        x = self.norm(x)
        x = self.relu(x)
        for i in x:
            self.edge_arr.append(i)

        return


if __name__ == '__main__':
    model = NUN(config.num_channels, config.num_classes)



    x = torch.randn(2, 3, 360, 360)
    print(model(x).shape)
