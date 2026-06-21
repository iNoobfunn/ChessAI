import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent dir so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ResidualBlock(nn.Module):
    """Standard 2-conv residual block (AlphaZero-style)."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class ChessMoveNet(nn.Module):
    """
    Residual CNN with two heads:
      - policy head → 4096 logits over 64×64 from-to moves (illegal masked at inference)
      - value  head → scalar in [-1, 1], expected game outcome for the side to move

    Input: (B, 18, 8, 8) board planes.
    """
    def __init__(self):
        super().__init__()
        C = config.CONV_CHANNELS

        # Input stem
        self.stem = nn.Sequential(
            nn.Conv2d(config.N_PLANES, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

        # Residual tower
        self.tower = nn.Sequential(*[ResidualBlock(C) for _ in range(config.N_RES_BLOCKS)])

        # Policy head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(C, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(32 * 64, config.OUTPUT_DIM)

        # Value head
        self.value_conv = nn.Sequential(
            nn.Conv2d(C, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(8 * 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        # Accept either (B, 18, 8, 8) or a flat (B, 1152) tensor.
        if x.dim() == 2:
            x = x.view(-1, config.N_PLANES, 8, 8)
        x = self.stem(x)
        x = self.tower(x)

        p = self.policy_conv(x).flatten(1)
        policy = self.policy_fc(p)                 # (B, 4096) raw logits

        v = self.value_conv(x).flatten(1)
        value = self.value_fc(v).squeeze(-1)       # (B,) in [-1, 1]

        return policy, value


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
    print(f"Parameters: {total:,}")
    x = torch.zeros(2, config.N_PLANES, 8, 8)
    p, v = m(x)
    print("policy", tuple(p.shape), "value", tuple(v.shape))
