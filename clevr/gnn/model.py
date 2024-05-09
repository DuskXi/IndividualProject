from loguru import logger
from rich.logging import RichHandler
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm.rich import tqdm
from timeit import default_timer as timer
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data as GData

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GNN(nn.Module):
    def __init__(self, num_classes, relation_hidden=256, tree_hidden=256, linear_hidden=512):
        super(GNN, self).__init__()
        self.relation_gcn = nn.ModuleDict({
            name: nn.ModuleList([
                GCNConv(15, relation_hidden),
                nn.ReLU(),
                GCNConv(relation_hidden, relation_hidden),
            ]) for name in ['left', 'right', 'front', 'behind']
        })
        self.question_tree_gcn_1 = GCNConv(39, tree_hidden)
        self.question_tree_gcn_2 = GCNConv(tree_hidden, tree_hidden)
        self.fc = nn.Sequential(
            nn.Linear(relation_hidden * 4 + tree_hidden, linear_hidden),
            nn.ReLU(),
            nn.Linear(linear_hidden, linear_hidden),
            nn.ReLU(),
            nn.Linear(linear_hidden, num_classes)
        )

    def forward_relation(self, relation: GData, direction: str):
        relation_x, relation_edge_index, relation_batch = relation.x, relation.edge_index, relation.batch
        module_list = self.relation_gcn[direction]
        for module in module_list:
            if isinstance(module, nn.ReLU):
                relation_x = torch.relu(relation_x)
            else:
                relation_x = module(relation_x, relation_edge_index)
        relation_features = global_mean_pool(relation_x, relation_batch)
        return relation_features

    def forward(self, relation_left: GData, relation_right: GData, relation_front: GData, relation_behind: GData, question_tree: GData):
        left = self.forward_relation(relation_left, 'left')
        right = self.forward_relation(relation_right, 'right')
        front = self.forward_relation(relation_front, 'front')
        behind = self.forward_relation(relation_behind, 'behind')
        relation_features = torch.cat([left, right, front, behind], dim=1)

        question_tree_x, question_tree_edge_index, question_tree_batch = question_tree.x, question_tree.edge_index, question_tree.batch
        question_tree_x = self.question_tree_gcn_1(question_tree_x, question_tree_edge_index)
        question_tree_x = torch.relu(question_tree_x)
        question_tree_x = self.question_tree_gcn_2(question_tree_x, question_tree_edge_index)
        question_tree_features = global_mean_pool(question_tree_x, question_tree_batch)

        x = torch.cat([relation_features, question_tree_features], dim=1)
        x = self.fc(x)
        return x


class CNN_GNN(nn.Module):
    def __init__(self, num_classes, relation_hidden=16, tree_hidden=64, linear_hidden=256):
        super(CNN_GNN, self).__init__()
        self.relation_gcn = nn.Sequential(
            nn.Conv2d(4, relation_hidden, kernel_size=3, padding=1),
            # nn.BatchNorm2d(relation_hidden),
            nn.ReLU(),
            nn.Conv2d(relation_hidden, relation_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(relation_hidden, relation_hidden, kernel_size=3, padding=1),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((5, 5))
        self.attribute_encoder = nn.Sequential(
            nn.Linear(480, linear_hidden),
            # nn.BatchNorm1d(linear_hidden),
            nn.ReLU(),
            nn.Linear(linear_hidden, linear_hidden),
            nn.ReLU(),
            nn.Linear(linear_hidden, linear_hidden)
        )
        self.question_tree_gcn = nn.ModuleList([
            GCNConv(50, tree_hidden),
            # nn.BatchNorm1d(tree_hidden),
            nn.GELU(),
            GCNConv(tree_hidden, tree_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            GCNConv(tree_hidden, tree_hidden)
        ])
        self.fc = nn.Sequential(
            nn.Linear(relation_hidden * 5 * 5 + linear_hidden + tree_hidden, linear_hidden),
            # nn.BatchNorm1d(linear_hidden),
            nn.GELU(),
            nn.Linear(linear_hidden, linear_hidden),
            nn.GELU(),
            nn.Linear(linear_hidden, linear_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(linear_hidden, num_classes)
        )

    def forward(self, relation: Tensor, attribute, question_tree: GData):
        relation_batch, direction_channel, w, h = relation.size()
        attribute_batch, num_object, attribute_features = attribute.size()
        relation_features = self.relation_gcn(relation)
        relation_features = self.avg_pool(relation_features)
        relation_features = relation_features.view(relation_batch, -1)
        attribute_features = self.attribute_encoder(attribute.view(attribute_batch, num_object * attribute_features))

        question_tree_x, question_tree_edge_index, question_tree_batch = question_tree.x, question_tree.edge_index, question_tree.batch
        for module in self.question_tree_gcn:
            if isinstance(module, GCNConv):
                question_tree_x = module(question_tree_x, question_tree_edge_index)
            else:
                question_tree_x = module(question_tree_x)
        question_tree_features = global_mean_pool(question_tree_x, question_tree_batch)

        # question_tree_x = self.question_tree_gcn_1(question_tree_x, question_tree_edge_index)
        # question_tree_x = torch.relu(question_tree_x)
        # question_tree_x = self.question_tree_gcn_2(question_tree_x, question_tree_edge_index)
        # question_tree_features = global_mean_pool(question_tree_x, question_tree_batch)

        # x = torch.cat([relation_features, attribute_features, torch.zeros(question_tree_features.shape, dtype=question_tree_features.dtype, device='cuda')], dim=1)
        x = torch.cat([relation_features, attribute_features, question_tree_features], dim=1)
        x = self.fc(x)
        return x
