# MCTNet - Transformer Conv Hybrid Model for Tabular Classification

This repository implements a hybrid deep learning model combining Transformer and Convolutional neural network (CNN) structures for tabular data classification tasks.

The framework is implemented in PyTorch and supports training, evaluation, and model checkpointing.

---

# 📁 Project Structure
MCTNet

├── checkpoints            # Saved best model weights (.pth)


├── data                   # Dataset directory (CSV files)


├── model                  # Model definition

└── MCTEnet.py

├── scaler                 # Saved StandardScaler


├── train.py                # Training script (main entry)


├── README.md               # Project documentation


---

# Features

- Hybrid Transformer + CNN architecture
- StandardScaler feature normalization
- Stratified train/test split
- PyTorch training pipeline
- Automatic best model saving
- Confusion matrix evaluation
- Reproducible results (fixed random seed)

---

# Installation

## 1. Create environment (recommended)

```bash
conda create -n mctnet python=3.9 -y
conda activate mctnet
pip install torch numpy pandas scikit-learn matplotlib joblib

