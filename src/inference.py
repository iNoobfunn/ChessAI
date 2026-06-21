import os, sys
import chess, torch
import torch.nn.functional as F

# Add parent dir so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.board_encoder import board_to_tensor, move_to_index, index_to_move
import config


def get_legal_move_mask(board: chess.Board) -> torch.Tensor:
    """Return a (4096,) bool tensor — True for each legal move index."""
    mask = torch.zeros(config.OUTPUT_DIM, dtype=torch.bool)
    for move in board.legal_moves:
        mask[move_to_index(move, board)] = True
    return mask


def policy_value(board: chess.Board, model, device: str = config.DEVICE):
    """Run the net once; return (masked legal probs over 4096, value scalar)."""
    tensor = board_to_tensor(board).unsqueeze(0).to(device)   # (1, 18, 8, 8)
    with torch.no_grad():
        logits, value = model(tensor)
    logits = logits.squeeze(0)
    mask = get_legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")
    return logits, value.item()


def predict_move(board: chess.Board, model, device: str = config.DEVICE,
                 temperature: float = 1.0):
    """
    Predict a move directly from the policy head (no search).

    temperature <= 0  → argmax (deterministic, strongest)
    temperature  > 0  → sample from the softmax (variety / exploration)

    Returns (move, value) where value ∈ [-1, 1] is the net's outcome estimate
    for the side to move.
    """
    logits, value = policy_value(board, model, device)

    if temperature and temperature > 0:
        probs = F.softmax(logits / temperature, dim=0)
        idx = torch.multinomial(probs, 1).item()
    else:
        idx = logits.argmax().item()

    move = index_to_move(idx, board)
    if move not in board.legal_moves:
        move = next(iter(board.legal_moves))
    return move, value


# ── Quick smoke test ───────────────────────────────────
if __name__ == "__main__":
    from src.model import load_model
    model = load_model(os.path.join(config.CHECKPOINT_DIR, "best.pt"))
    board = chess.Board()
    for _ in range(5):
        move, val = predict_move(board, model)
        print("Predicted:", move, "(" + board.san(move) + f")  value={val:+.2f}")
        board.push(move)
