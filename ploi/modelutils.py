import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer


class EdgeModel(nn.Module):
    def __init__(self, n_features, n_edge_features, n_hidden, dropout=0.0):
        super(EdgeModel, self).__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * n_features + n_edge_features, n_hidden),
            nn.ReLU(),
            nn.LayerNorm(n_hidden),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(n_hidden, n_hidden),
        )

    def forward(self, src, dst, edge_attr, u=None, batch=None):
        # src, dst: [E, F_x], where E is num edges, F_x is node-feature dimensionality
        # edge_attr: [E, F_e], where E is num edges, F_e is edge-feature dimensionality
        out = torch.cat([src, dst, edge_attr], dim=1)
        return self.edge_mlp(out)


class NodeModel(nn.Module):
    def __init__(self, n_features, n_edge_features, n_hidden, n_targets, dropout=0.0):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(n_features + n_hidden, n_hidden),
            nn.ReLU(),
            nn.LayerNorm(n_hidden),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(n_hidden, n_hidden),
        )
        self.node_mlp_2 = nn.Sequential(
            nn.Linear(n_features + n_hidden, n_hidden),
            nn.ReLU(),
            nn.LayerNorm(n_hidden),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(n_hidden, n_targets),
        )

    def forward(self, x, edge_idx, edge_attr, u=None, batch=None):
        row, col = edge_idx
        out = torch.cat([x[col], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, row, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)


class GraphNetwork(nn.Module):
    def __init__(self, n_features, n_edge_features, n_hidden, dropout=0.0):
        super(GraphNetwork, self).__init__()
        self.meta_layer_1 = self.build_meta_layer(
            n_features, n_edge_features, n_hidden, n_hidden, dropout=dropout
        )
        self.meta_layer_2 = self.build_meta_layer(
            n_hidden, n_hidden, n_hidden, n_hidden, dropout=dropout
        )
        self.meta_layer_3 = self.build_meta_layer(
            n_hidden, n_hidden, n_hidden, 1, dropout=dropout
        )

    def build_meta_layer(
        self, n_features, n_edge_features, n_hidden, n_targets, dropout=0.0
    ):
        return MetaLayer(
            edge_model=EdgeModel(
                n_features, n_edge_features, n_hidden, dropout=dropout
            ),
            node_model=NodeModel(
                n_features, n_edge_features, n_hidden, n_targets, dropout=dropout
            ),
        )

    def forward(self, x, edge_idx, edge_attr, u=None, batch=None):
        x, edge_attr, _ = self.meta_layer_1(x, edge_idx, edge_attr)
        x, edge_attr, _ = self.meta_layer_2(x, edge_idx, edge_attr)
        x, edge_attr, _ = self.meta_layer_3(x, edge_idx, edge_attr)
        return x
