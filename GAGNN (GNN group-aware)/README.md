# GAGNN: Group-Aware Graph Neural Network for Anti-Money Laundering

## 📌 Project Overview
GAGNN is a sophisticated deep learning pipeline designed to detect money laundering activities in financial transaction data. It utilizes **Graph Neural Networks (GNNs)** to capture complex relationships between accounts and transactions, moving beyond traditional rule-based systems to identify "Gangs" or laundering groups.

The project is specifically optimized for the **IBM AML Dataset** and focuses on high-recall detection to identify criminal networks effectively while balancing precision through advanced regularization and combined loss functions.

---

## 🚀 Key Features
- **Group-Aware Aggregation**: Clusters suspicious transactions into "Gangs" to track coordinated laundering groups.
- **eMRF Energy Refinement**: Uses an enhanced Markov Random Field layer to refine node embeddings based on topological and semantic similarity.
- **Cost-Sensitive Learning**: Implements a weighted loss function (1:40 ratio) to prioritize the detection of rare laundering events.
- **Multi-Head Attention (GAT)**: Employs 8 attention heads to simultaneously analyze different "red-flag" patterns (e.g., rapid outflow, cycling).
- **Mixed-Precision Training**: Utilizes PyTorch AMP for memory-efficient and fast GPU training.
- **Interpretability**: Includes ablation-based feature importance analysis and hard-sample error tracking.

---

## 🏗️ Technical Architecture
1. **Data Loading & Preprocessing**: Loads IBM AML data, performs log-scaling on financial amounts, and one-hot encodes categorical features.
2. **Graph Construction**: Builds a `MultiDiGraph` where nodes are accounts and edges are transactions. Enriches nodes with features like degree variance, fraud rates, and flow statistics.
3. **GAGNN Model**:
   - **GAT Encoder**: Captures local neighborhood information.
   - **eMRF Layer**: Refines features using Jaccard similarity and cosine similarity.
   - **Edge MLP**: Predicts the likelihood of a transaction being part of a laundering scheme.
   - **Group Scorer**: Evaluates the collective risk of detected transaction clusters (Gangs).
4. **Trainer & Evaluator**: Manages mixed-precision training loops, early stopping, and comprehensive metric calculation (Precision, Recall, F1, AUC).

---

## 📂 Project Structure
- `main_pipeline.py`: The central orchestration script to run the full end-to-end training and evaluation.
- `model_core.py`: Definitions for the `GAGNN_Model` and its sub-components (GAT, eMRF, Gang clustering).
- `graph_builder.py`: Logic for converting tabular data into enriched graph structures suitable for GNNs.
- `data_loader.py`: Specialized dataset loading and feature engineering for the IBM AML schema.
- `trainer.py`: Gradient-scaling trainer implementation with mixed-precision support.
- `loss_functions.py`: Custom combined loss (Edge + Group) with cost-sensitive weighting.
- `evaluation.py`: Modules for metrics calculation, early stopping, and visualization (plots).

---

## 💻 Environment Requirements
This project requires **Python 3.12.0** and the following core libraries:

| Library | Version | Description |
| :--- | :--- | :--- |
| `torch` | 2.5.0 | Core tensor and deep learning framework |
| `torch-geometric` | 2.7.0 | Graph Neural Network extensions |
| `networkx` | 3.6.1 | Graph algorithms for Gang clustering |
| `pandas` | 3.0.1 | Data manipulation and analysis |
| `numpy` | 2.2.6 | Numerical computing |
| `scikit-learn` | 1.4.2 | Metrics and data preprocessing |
| `scipy` | 1.13.0 | Scientific computing support |

---

## 📥 Installation & Setup
1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd GAGNN
   ```
2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121  # Adjust for CUDA version
   pip install torch-geometric==2.7.0 torch-scatter torch-sparse
   pip install pandas==3.0.1 numpy==2.2.6 scikit-learn==1.4.2 networkx==3.6.1 matplotlib tqdm
   ```

---

## 🏃 Usage
To start the training and evaluation pipeline, run:
```bash
python GAGNN/main_pipeline.py
```
**Configuration Notes**:
- You can adjust `DATASET_TO_RUN` in `main_pipeline.py` to point to your specific IBM dataset file.
- Hyperparameters like `CLASS_WEIGHTS`, `BATCH_SIZE`, and `EPOCHS` are at the top of `main_pipeline.py`.

---

## 📊 Results & Visualization
Upon completion, the pipeline generates several diagnostic files in the root directory:
- `confusion_matrix.png`: Visualizes false positives vs. true positives.
- `pr_curve.png`: Shows the trade-off between Precision and Recall.
- `train_test_gap.png`: Monitors training loss and recall trends across epochs.
- `feature_importance.png`: Displays which node features (e.g., In-degree, Fraud-rate) the model relies on most.
- `hard_samples.txt`: Log of false negatives and positives for forensic review.
- `best_model.pth`: The saved model weights with the highest F1 score.

---

## 📋 License
MIT
