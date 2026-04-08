import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GAGNN_Model(nn.Module):
    def __init__(self, in_channels, edge_in_channels, hidden_channels=64, heads=8):
        super().__init__()
        
        # 1. GAT Encoder: Learns local patterns
        self.gat = GATConv(
            in_channels=in_channels, 
            out_channels=hidden_channels,
            heads=heads, 
            concat=True, 
            edge_dim=edge_in_channels,
            dropout=0.2
        )
        self.gat_out_dim = hidden_channels * heads
        
        # 2. Refinement & Detection MLP
        # Consolidates node and edge features into a single prediction path
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.gat_out_dim * 2 + edge_in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # Raw logits for CrossEntropy
        )
        
        # 3. Group Risk Scorer (Lean: uses global pooling)
        self.group_scorer = nn.Sequential(
            nn.Linear(self.gat_out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_attr):
        # Local Encoding
        h = self.gat(x, edge_index, edge_attr=edge_attr)
        h = F.leaky_relu(h, 0.2)
        
        # Edge Logic: Cat source node, target node, and edge attributes
        row, col = edge_index
        edge_feats = torch.cat([h[row], h[col], edge_attr], dim=1)
        edge_logits = self.edge_mlp(edge_feats)
        
        # Group Logic: Lean pooling (Treats the whole batch as a potential group/gang)
        # In a real scenario, this would be node-level clustering, but for lean 
        # we pool over the entire subgraph batch.
        Z_g = global_mean_pool(h, torch.zeros(h.size(0), dtype=torch.long, device=x.device))
        group_risk = self.group_scorer(Z_g).squeeze(-1)
        
        return edge_logits, group_risk

if __name__ == "__main__":
    print("Lean GAGNN Check...")
    m = GAGNN_Model(in_channels=9, edge_in_channels=5)
    x = torch.randn((100, 9))
    ei = torch.randint(0, 100, (2, 500))
    ea = torch.randn((500, 5))
    logits, risk = m(x, ei, ea)
    print(f"Logits shape: {logits.shape}, Risk: {risk.item():.4f}")
