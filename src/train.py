import os, sys, torch, torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Add parent dir so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.model import ChessMoveNet, save_model
from src.dataset import get_dataloaders


def masked_topk_accuracy(logits, targets, weights, k=1):
    """Top-k accuracy computed only over winner-move (weight==1) rows."""
    sel = weights > 0
    if sel.sum() == 0:
        return 0.0
    topk = logits[sel].topk(k, dim=1).indices
    correct = topk.eq(targets[sel].unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


def train(resume: bool = False):
    device = config.DEVICE
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    train_loader, val_loader = get_dataloaders(
        os.path.join(config.DATA_PROC_DIR, "dataset.npz")
    )

    model = ChessMoveNet().to(device)

    start_epoch  = 1
    best_val_acc = 0.0
    ckpt = os.path.join(config.CHECKPOINT_DIR, "best.pt")
    if resume and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        start_epoch  = int(state.get("epoch", 0)) + 1
        best_val_acc = float(state.get("val_acc", 0.0))
        print(f"Resuming from epoch {start_epoch} (best top-1 so far {best_val_acc:.3f})")

    # Optimizer state isn't checkpointed, so re-create it and fast-forward the LR
    # schedule to match how many epochs have already elapsed.
    decay_steps = (start_epoch - 1) // config.LR_STEP
    lr = config.LR * (config.LR_GAMMA ** decay_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=config.LR_STEP, gamma=config.LR_GAMMA)

    history = {"train_loss": [], "val_top1": [], "val_top5": [], "val_vmae": []}

    for epoch in range(start_epoch, config.EPOCHS + 1):
        # ── Train ─────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for X, y, v, pw in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            X, y, v, pw = X.to(device), y.to(device), v.to(device), pw.to(device)
            optimizer.zero_grad()
            policy, value = model(X)

            # Policy: cross-entropy weighted to winner moves only (#5)
            ce = F.cross_entropy(policy, y, reduction="none")
            denom = pw.sum().clamp(min=1.0)
            policy_loss = (ce * pw).sum() / denom

            # Value: MSE over every position (#1)
            value_loss = F.mse_loss(value, v)

            loss = policy_loss + config.VALUE_LOSS_WEIGHT * value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # ── Validate ──────────────────────────────────────
        model.eval()
        top1_list, top5_list, vmae_list = [], [], []
        with torch.no_grad():
            for X, y, v, pw in val_loader:
                X, y, v, pw = X.to(device), y.to(device), v.to(device), pw.to(device)
                policy, value = model(X)
                top1_list.append(masked_topk_accuracy(policy, y, pw, k=1))
                top5_list.append(masked_topk_accuracy(policy, y, pw, k=5))
                vmae_list.append((value - v).abs().mean().item())

        val_top1 = sum(top1_list) / len(top1_list)
        val_top5 = sum(top5_list) / len(top5_list)
        val_vmae = sum(vmae_list) / len(vmae_list)
        scheduler.step()

        history["train_loss"].append(avg_loss)
        history["val_top1"].append(val_top1)
        history["val_top5"].append(val_top5)
        history["val_vmae"].append(val_vmae)

        print(f"Epoch {epoch:02d} | loss {avg_loss:.4f} | top1 {val_top1:.3f} | "
              f"top5 {val_top5:.3f} | value MAE {val_vmae:.3f}")

        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            save_model(model, ckpt, epoch, val_top1)
            print(f"  -> Saved best checkpoint ({val_top1:.3f})")

    _save_curves(history)
    print("Training complete. Best top-1:", best_val_acc)


def _save_curves(history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    ax1.plot(history["train_loss"], label="train loss"); ax1.set_title("Loss"); ax1.legend()
    ax2.plot(history["val_top1"], label="top-1")
    ax2.plot(history["val_top5"], label="top-5"); ax2.set_title("Policy accuracy"); ax2.legend()
    ax3.plot(history["val_vmae"], label="value MAE"); ax3.set_title("Value MAE"); ax3.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "training_curves.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    resume = "--resume" in sys.argv
    train(resume=resume)
