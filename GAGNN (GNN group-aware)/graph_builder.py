import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from sklearn.preprocessing import StandardScaler

def build_multidigraph(df):
    """Refined Graph Builder: Vectorized calculations for speed."""
    # 1. Account Mapping
    unique_acc = np.unique(df[['Account', 'Account.1']].values)
    acc_map = {acc: i for i, acc in enumerate(unique_acc)}
    
    src = df['Account'].map(acc_map).values
    dst = df['Account.1'].map(acc_map).values
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # 2. Node Features (Lean)
    num_nodes = len(unique_acc)
    nodes = pd.DataFrame(index=unique_acc)
    
    # Fast grouping for stats
    out_stats = df.groupby('Account')['Amount_Received'].agg(['count', 'sum', 'mean'])
    in_stats = df.groupby('Account.1')['Amount_Received'].agg(['count', 'sum', 'mean'])
    
    node_feats = nodes.join(out_stats, rsuffix='_out').join(in_stats, rsuffix='_in').fillna(0)
    
    # Calculate degree variance simplified
    total_deg = node_feats['count'] + node_feats['count_in']
    node_feats['deg_var'] = (node_feats['count'] - node_feats['count_in']).abs() / (total_deg + 1)
    
    # Normalize
    scaler = StandardScaler()
    x = torch.from_numpy(scaler.fit_transform(node_feats.values)).float()

    # 3. Edge Features
    exclude = ['Timestamp', 'From_Bank', 'To_Bank', 'Account', 'Account.1', 'Is_Laundering', 'Receiving_Currency', 'Payment_Currency', 'Payment_Format']
    feat_cols = [c for c in df.columns if c not in exclude]
    edge_attr = torch.tensor(df[feat_cols].values, dtype=torch.float)
    y = torch.tensor(df['Is_Laundering'].values, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

def get_neighbor_loader(data, batch_size=128, num_neighbors=[15, 10]):
    return NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size, directed=True, shuffle=True)
