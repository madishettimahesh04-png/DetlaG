import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GATConv,
    SAGEConv,
    global_mean_pool
)


# =====================================================
# GNN BRANCH
# =====================================================

class GNN(nn.Module):

    def __init__(
        self,
        hidden_dim=128,
        heads=2,
        dropout=0.1288290982877934
    ):

        super().__init__()

        self.gat1 = GATConv(
            in_channels=9,
            out_channels=hidden_dim,
            heads=heads
        )

        self.sage1 = SAGEConv(
            hidden_dim * heads,
            hidden_dim
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = self.gat1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.dropout(x)

        x = self.sage1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = global_mean_pool(
            x,
            batch
        )

        return x


# =====================================================
# COMPLETE MODEL
# =====================================================

class Model(nn.Module):

    def __init__(
        self,
        desc_dim=155,
        hidden_dim=128,
        heads=2,
        dropout=0.1288290982877934,
        desc_hidden=256
    ):

        super().__init__()

        # -------------------------
        # Graph encoder
        # -------------------------

        self.gnn = GNN(
            hidden_dim=hidden_dim,
            heads=heads,
            dropout=dropout
        )

        # -------------------------
        # Descriptor encoder
        # -------------------------

        self.desc_net = nn.Sequential(

            nn.Linear(
                desc_dim,
                desc_hidden
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                desc_hidden,
                64
            ),

            nn.ReLU()
        )

        # -------------------------
        # Final regressor
        # -------------------------

        self.fc = nn.Sequential(

            nn.Linear(
                hidden_dim + 64,
                128
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                128,
                1
            )
        )

    def forward(
        self,
        g1,
        g2,
        descriptors
    ):

        # -------------------------
        # Solute graph
        # -------------------------

        h1 = self.gnn(g1)

        # -------------------------
        # Solvent graph
        # -------------------------

        h2 = self.gnn(g2)

        # -------------------------
        # Combine graphs
        # -------------------------

        graph_feat = h1 + h2

        # -------------------------
        # Descriptor branch
        # -------------------------

        desc_feat = self.desc_net(
            descriptors
        )

        # -------------------------
        # Fusion
        # -------------------------

        x = torch.cat(
            [
                graph_feat,
                desc_feat
            ],
            dim=1
        )

        out = self.fc(x)

        return out.squeeze(-1)
