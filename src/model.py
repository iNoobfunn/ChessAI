import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent dir so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class ChessMoveNet(nn.Module):
    """
    MLP: 768 → 1024 → 512 → 256 → 4096
    Predicts move probability distribution over all 64×64 from-to pairs.
    Illegal moves are masked at inference time (see inference.py).
    """
    def __init__(self):
        super().__init__()
        dims = [config.INPUT_DIM] + config.HIDDEN_DIMS + [config.OUTPUT_DIM]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:          # no activation on final layer
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(config.DROPOUT))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # raw logits (4096,)


def load_model(checkpoint_path: str, device: str = config.DEVICE) -> ChessMoveNet:
    model = ChessMoveNet().to(device)
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
    else:
        print(f"WARNING: No checkpoint found at {checkpoint_path}, using random weights")
    model.eval()
    return model


def save_model(model: ChessMoveNet, path: str, epoch: int, val_acc: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(),
                "epoch": epoch, "val_acc": val_acc}, path)


# Count parameters
if __name__ == "__main__":
    m = ChessMoveNet()
    total = sum(p.numel() for p in m.parameters())
    print(f"Parameters: {total:,}")  # expect ~2.5M
