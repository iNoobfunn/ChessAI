import os, sys, torch, torch.nn as nn
from tqdm import tqdm

# Add parent dir so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.model import ChessMoveNet, save_model
from src.dataset import get_dataloaders

def top_k_accuracy(logits, targets, k=5):
    topk = logits.topk(k, dim=1).indices
    correct = topk.eq(targets.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()

def train():
    device = config.DEVICE
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    train_loader, val_loader = get_dataloaders(
        os.path.join(config.DATA_PROC_DIR, "dataset.npz")
    )

    model     = ChessMoveNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=config.LR_STEP, gamma=config.LR_GAMMA)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    history = {"train_loss": [], "val_top1": [], "val_top5": []}

    for epoch in range(1, config.EPOCHS + 1):
        # ── Train ─────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # ── Validate ──────────────────────────────────────
        model.eval()
        top1_list, top5_list = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                top1_list.append(top_k_accuracy(logits, y_batch, k=1))
                top5_list.append(top_k_accuracy(logits, y_batch, k=5))

        val_top1 = sum(top1_list) / len(top1_list)
        val_top5 = sum(top5_list) / len(top5_list)
        scheduler.step()

        history["train_loss"].append(avg_loss)
        history["val_top1"].append(val_top1)
        history["val_top5"].append(val_top5)

        print(f"Epoch {epoch:02d} | loss {avg_loss:.4f} | top1 {val_top1:.3f} | top5 {val_top5:.3f}")

        # Save best checkpoint
        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            ckpt = os.path.join(config.CHECKPOINT_DIR, "best.pt")
            save_model(model, ckpt, epoch, val_top1)
            print(f"  -> Saved best checkpoint ({val_top1:.3f})")

    # Save training curves
    _save_curves(history)
    print("Training complete. Best top-1:", best_val_acc)

def _save_curves(history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history["train_loss"], label="train loss"); ax1.set_title("Loss"); ax1.legend()
    ax2.plot(history["val_top1"],  label="top-1")
    ax2.plot(history["val_top5"],  label="top-5"); ax2.set_title("Accuracy"); ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "training_curves.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    train()
