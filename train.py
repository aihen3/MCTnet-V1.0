import os
import time
import random
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from model.MCTnet import TransformerConvClassifier


# =========================
# Config
# =========================
class Config:
    csv_path = "./data/dataset.csv"
    feature_start = 0
    feature_end = 7
    label_col = -1

    batch_size = 32
    lr = 1e-3
    weight_decay = 1e-4
    epochs = 100
    test_size = 0.2
    seed = 42

    scaler_path = "./scaler/feature_scaler.pkl"
    model_dir = "./checkpoints"


cfg = Config()
os.makedirs(cfg.model_dir, exist_ok=True)
os.makedirs("./scaler", exist_ok=True)


# =========================
# Reproducibility
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(cfg.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# =========================
# Load Data
# =========================
data = pd.read_csv(cfg.csv_path)

X_raw = data.iloc[:, cfg.feature_start:cfg.feature_end].values
y_raw = data.iloc[:, cfg.label_col].values.astype(np.int64)


# =========================
# Train/Test Split
# =========================
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw,
    y_raw,
    test_size=cfg.test_size,
    random_state=cfg.seed,
    stratify=y_raw
)


# =========================
# Scaling (NO leakage)
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

joblib.dump(scaler, cfg.scaler_path)
print("Scaler saved ->", cfg.scaler_path)


# =========================
# Dataset
# =========================
train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)

test_ds = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.long)
)

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)


# =========================
# Model
# =========================
model = TransformerConvClassifier().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.lr,
    weight_decay=cfg.weight_decay
)

criterion = nn.CrossEntropyLoss()


# =========================
# Train Loop
# =========================
best_acc = 0.0

for epoch in range(1, cfg.epochs + 1):
    model.train()
    losses = []
    start = time.time()

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    # =========================
    # Eval
    # =========================
    model.eval()
    preds_all, labels_all = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(yb.numpy())

    acc = accuracy_score(labels_all, preds_all)
    avg_loss = np.mean(losses)

    # save best model
    if acc > best_acc:
        best_acc = acc
        torch.save(
            model.state_dict(),
            os.path.join(cfg.model_dir, "best_model.pth")
        )

    if epoch % 10 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {avg_loss:.4f} | "
            f"Acc: {acc:.4f} | "
            f"Best: {best_acc:.4f} | "
            f"Time: {time.time() - start:.2f}s"
        )


# =========================
# Final Test
# =========================
model.load_state_dict(torch.load(os.path.join(cfg.model_dir, "best_model.pth")))
model.eval()

preds_all, labels_all = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb)

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        preds_all.extend(preds)
        labels_all.extend(yb.numpy())

acc = accuracy_score(labels_all, preds_all)
cm = confusion_matrix(labels_all, preds_all)

print("\nFinal Test Accuracy:", acc)
print("Confusion Matrix:\n", cm)
print("Best model saved in:", cfg.model_dir)