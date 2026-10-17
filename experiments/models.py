"""The three compared models. All are two layers deep with the same dropout placement:

    dropout(x) -> layer 1 -> activation -> dropout -> layer 2 -> logits

Differences are limited to the layer type (and GAT's standard ELU activation and attention heads):
  * MLP  - nn.Linear layers. forward() takes node features only, so it cannot see the graph.
  * GCN  - GCNConv (symmetric-normalised neighbourhood averaging, self-loops added by the layer).
  * GAT  - GATConv with `heads` attention heads concatenated in layer 1 and one head in layer 2;
           the same dropout rate is applied to the attention coefficients, as in Velickovic et al. (2018).
"""
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv


class MLP(nn.Module):
    uses_graph = False

    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, self.dropout, self.training)
        return self.lin2(x)


class GCN(nn.Module):
    uses_graph = True

    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, self.training)
        return self.conv2(x, edge_index)


class GAT(nn.Module):
    uses_graph = True

    def __init__(self, in_dim, hidden, out_dim, dropout, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden * heads, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, self.training)
        return self.conv2(x, edge_index)


def build_model(name, in_dim, out_dim, hparams):
    if name == "mlp":
        return MLP(in_dim, hparams["hidden"], out_dim, hparams["dropout"])
    if name == "gcn":
        return GCN(in_dim, hparams["hidden"], out_dim, hparams["dropout"])
    if name == "gat":
        return GAT(in_dim, hparams["hidden"], out_dim, hparams["dropout"], heads=hparams.get("heads", 8))
    raise ValueError(f"unknown model: {name}")
