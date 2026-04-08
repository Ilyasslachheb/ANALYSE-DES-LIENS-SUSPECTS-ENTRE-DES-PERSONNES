import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, PrecisionRecallDisplay

class VisualizationManager:
    def __init__(self, output_dir='.'):
        self.output_dir = output_dir

    def plot_results(self, y_true, y_pred, y_prob):
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5,5))
        plt.imshow(cm, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()

        # PR Curve
        PrecisionRecallDisplay.from_predictions(y_true, y_prob)
        plt.title('PR Curve')
        plt.savefig(os.path.join(self.output_dir, 'pr_curve.png'))
        plt.close()

class EarlyStopping:
    def __init__(self, patience=3, save_path='best_model.pth'):
        self.patience, self.save_path = patience, save_path
        self.counter, self.best_score = 0, None
        self.early_stop = False

    def __call__(self, score, model):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            torch.save(model.state_dict(), self.save_path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
