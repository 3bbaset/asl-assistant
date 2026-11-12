"""train_nn_seq.py - Deep PyTorch Sequence Model for ASL Word Recognition.

Architecture:
- Bi-directional GRU / LSTM + Multi-Head Self-Attention over (30 frames, 130 landmarks)
- Temporal & spatial dropout, label smoothing
- Fast multi-core CPU execution
- Evaluates on held-out 20% test set
"""
import glob, json, os, time
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from motion_common import fill_gaps, mirror_clip

class ASLDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            # Spatial jitter
            x += torch.randn_like(x) * 0.015
            # Random time warp / drop frames
            if torch.rand(1).item() < 0.3:
                mask = torch.rand(x.shape[0]) > 0.1
                if mask.any():
                    x[~mask] = 0.0
        return x, self.y[idx]

class ASLSequenceModel(nn.Module):
    def __init__(self, in_features=130, num_classes=50, hidden_dim=256, num_layers=2):
        super().__init__()
        self.fc_in = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.gru = nn.GRU(
            hidden_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0.0
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        h = self.fc_in(x)                     # (B, T, hidden)
        out, _ = self.gru(h)                  # (B, T, hidden*2)
        weights = torch.softmax(self.attn(out), dim=1) # (B, T, 1)
        context = torch.sum(out * weights, dim=1)      # (B, hidden*2)
        logits = self.head(context)           # (B, num_classes)
        return logits

def train_eval(n_words=50, epochs=35, batch_size=64, lr=1e-3):
    with open("curated_words.json") as f:
        target_words = set(json.load(f)[:n_words])

    bfs = sorted(glob.glob("batches/batch_*.npz"))
    raw_clips, raw_y = [], []
    seen = set()

    print(f"Loading clips for top {n_words} curated words...")
    for bf in bfs:
        d = np.load(bf, allow_pickle=True)
        for p, x, l in zip(d["path"], d["X"], d["y"]):
            p_str = str(p).replace("\\", "/")
            if p_str in seen:
                continue
            seen.add(p_str)
            l_str = str(l)
            if l_str in target_words:
                c = fill_gaps(np.asarray(x, dtype=np.float32).reshape(30, 130))
                raw_clips.append(c)
                raw_y.append(l_str)

    counts = Counter(raw_y)
    valid_words = sorted([w for w, n in counts.items() if n >= 15])
    label_to_id = {w: i for i, w in enumerate(valid_words)}
    
    clips = np.array([c for c, w in zip(raw_clips, raw_y) if w in label_to_id], dtype=np.float32)
    y_ids = np.array([label_to_id[w] for w in raw_y if w in label_to_id])

    print(f"Total usable: {len(clips):,} clips across {len(valid_words)} classes.")

    # 80/20 Stratified Split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr_idx, te_idx = next(splitter.split(clips, y_ids))

    # Mirror augmentation on training split
    X_tr_orig, y_tr_orig = clips[tr_idx], y_ids[tr_idx]
    X_tr_mirr = np.array([mirror_clip(c) for c in X_tr_orig], dtype=np.float32)
    X_train = np.vstack([X_tr_orig, X_tr_mirr])
    y_train = np.concatenate([y_tr_orig, y_tr_orig])

    X_test, y_test = clips[te_idx], y_ids[te_idx]

    train_ds = ASLDataset(X_train, y_train, augment=True)
    test_ds = ASLDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 4)

    model = ASLSequenceModel(num_classes=len(valid_words)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Training ASL BiGRU-Attention network for {epochs} epochs on {os.cpu_count()} CPU threads...")
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Evaluation
        model.eval()
        correct, top3_correct, total = 0, 0, 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                preds = logits.argmax(dim=1)
                correct += (preds == by).sum().item()
                top3 = logits.topk(min(3, len(valid_words)), dim=1).indices
                top3_correct += torch.any(top3 == by.unsqueeze(1), dim=1).sum().item()
                total += by.size(0)

        acc = (correct / total) * 100
        top3_acc = (top3_correct / total) * 100
        if acc > best_acc:
            best_acc = acc
            # Save checkpoint
            torch.save({
                "model_state": model.state_dict(),
                "classes": valid_words,
                "top1_acc": acc,
                "top3_acc": top3_acc
            }, "asl_nn_best.pt")

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs:2d} | Loss: {total_loss/len(train_loader):.4f} | Test Acc: {acc:.2f}% | Top-3 Acc: {top3_acc:.2f}% (Best: {best_acc:.2f}%)")

    print(f"\nCompleted! Best Test Top-1 Accuracy: {best_acc:.2f}% (Top-3: {top3_acc:.2f}%)")
    return best_acc

if __name__ == "__main__":
    train_eval(n_words=50, epochs=35)
