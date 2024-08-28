import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCN_FedTAD(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim,num_layers, dropout):
        super(GCN_FedTAD, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        embedding = F.dropout(x, p=self.dropout)
        x = self.conv2(embedding, edge_index)
        return F.log_softmax(x, dim=1)

    def rep_forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv2(x, edge_index)
        return x