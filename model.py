import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GATConv,
    SAGEConv,
    global_mean_pool
)


# ==========================================================
# GNN BLOCK
# ==========================================================
class GNN(nn.Module):

    def __init__(
        self,
        hidden_dim,
        heads,
        dropout
    ):
        super().__init__()

        self.gat1 = GATConv(
            in_channels=9,
            out_channels=hidden_dim,
            heads=heads,
            dropout=dropout
        )

        self.sage1 = SAGEConv(
            hidden_dim * heads,
            hidden_dim
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):

        x = self.gat1(
            data.x,
            data.edge_index
        )

        x = F.relu(x)

        x = self.dropout(x)

        x = self.sage1(
            x,
            data.edge_index
        )

        x = F.relu(x)

        x = global_mean_pool(
            x,
            data.batch
        )

        return x


# ==========================================================
# FULL MODEL
# ==========================================================
class Model(nn.Module):

    def __init__(
        self,
        desc_dim,
        hidden_dim,
        heads,
        dropout,
        desc_hidden
    ):
        super().__init__()

        self.gnn = GNN(
            hidden_dim=hidden_dim,
            heads=heads,
            dropout=dropout
        )

        # Descriptor network
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

        # Final prediction network
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
        d
    ):

        # Solute graph embedding
        g1_emb = self.gnn(g1)

        # Solvent graph embedding
        g2_emb = self.gnn(g2)

        # Interaction
        graph_features = g1_emb * g2_emb

        # Descriptor embedding
        desc_features = self.desc_net(d)

        # Concatenate
        x = torch.cat(
            [
                graph_features,
                desc_features
            ],
            dim=1
        )

        # Prediction
        out = self.fc(x)

        return out.squeeze()