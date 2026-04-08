import torch
import torch.nn as nn

class GAGNNCombinedLoss(nn.Module):
    """Simple combined loss for edge detection and group risk."""
    def __init__(self, class_weights=(1.0, 40.0)):
        super().__init__()
        self.edge_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
        self.group_loss = nn.BCELoss()

    def forward(self, edge_logits, edge_labels, group_score, group_label):
        l_edge = self.edge_loss(edge_logits, edge_labels)
        l_group = self.group_loss(group_score, group_label)
        return l_edge + 0.1 * l_group
