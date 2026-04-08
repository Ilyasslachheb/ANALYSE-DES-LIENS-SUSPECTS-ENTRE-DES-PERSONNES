import torch
from sklearn.metrics import recall_score, f1_score

class GAGNNTrainer:
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scaler = torch.amp.GradScaler('cuda' if 'cuda' in device.type else 'cpu')

    def train_step(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        data = data.to(self.device)
        
        with torch.amp.autocast(device_type=self.device.type):
            logits, group_risk = self.model(data.x, data.edge_index, data.edge_attr)
            # Simple group label: True if any edge in batch is laundering
            group_label = (data.y.max() > 0).float().unsqueeze(0)
            loss = self.loss_fn(logits, data.y, group_risk, group_label)
            
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        y_true, y_pred, y_prob = [], [], []
        for data in loader:
            data = data.to(self.device)
            logits, _ = self.model(data.x, data.edge_index, data.edge_attr)
            probs = torch.softmax(logits, dim=1)[:, 1]
            y_true.append(data.y.cpu())
            y_pred.append(logits.argmax(dim=1).cpu())
            y_prob.append(probs.cpu())
            
        y_true, y_pred, y_prob = torch.cat(y_true).numpy(), torch.cat(y_pred).numpy(), torch.cat(y_prob).numpy()
        return {
            'recall': recall_score(y_true, y_pred) * 100,
            'f1': f1_score(y_true, y_pred) * 100,
            'labels': y_true, 'preds': y_pred, 'probs': y_prob
        }
