import os
import torch
from data_loader import load_dataset, preprocess_data
from graph_builder import build_multidigraph, get_neighbor_loader
from model_core import GAGNN_Model
from loss_functions import GAGNNCombinedLoss
from trainer import GAGNNTrainer
from evaluation import EarlyStopping, VisualizationManager

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Lean GAGNN on {device}...")

    # 1. Data
    data_path = r'c:\Users\HP\Downloads\my_project\GAGNN\HI-Small_Trans.csv'
    df = load_dataset(data_path)
    df = preprocess_data(df)
    data = build_multidigraph(df)

    # 2. Model & Trainer
    loader = get_neighbor_loader(data, batch_size=4096)
    model = GAGNN_Model(in_channels=data.x.shape[1], edge_in_channels=data.edge_attr.shape[1])
    trainer = GAGNNTrainer(model, torch.optim.AdamW(model.parameters(), lr=5e-4), GAGNNCombinedLoss(), device)
    
    stopper = EarlyStopping()
    viz = VisualizationManager()

    # 3. Train
    for epoch in range(1, 11): # 10 epochs for lean demo
        avg_loss = 0
        for i, batch in enumerate(loader):
            avg_loss += trainer.train_step(batch)
            if i >= 4: break # 5 batches per epoch
        
        metrics = trainer.evaluate([batch])
        print(f"Epoch {epoch} | Loss: {avg_loss/5:.4f} | F1: {metrics['f1']:.2f}%")
        stopper(metrics['f1'], model)
        if stopper.early_stop: break

    # 4. Final
    final = trainer.evaluate(loader)
    viz.plot_results(final['labels'], final['preds'], final['probs'])
    print(f"Final Recall: {final['recall']:.2f}% | F1: {final['f1']:.2f}%")

if __name__ == "__main__":
    main()
