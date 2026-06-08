import os, sys
import chess, torch
import torch.nn.functional as F

# Add parent dir so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.board_encoder import board_to_tensor, move_to_index, index_to_move
import config

def get_legal_move_mask(board: chess.Board) -> torch.Tensor:
    """Return a (4096,) bool tensor — True for each legal move index."""
    mask = torch.zeros(4096, dtype=torch.bool)
    for move in board.legal_moves:
        idx = move_to_index(move, board)
        mask[idx] = True
    return mask


def predict_move(board: chess.Board, model, device: str = config.DEVICE,
                  temperature: float = 1.0) -> chess.Move:
    """
    Predict the best move for the current board position.

    Args:
        board:       chess.Board in any position
        model:       loaded ChessMoveNet (in eval mode)
        temperature: 1.0 = argmax-like, >1.0 = more random (exploration)

    Returns:
        A legal chess.Move
    """
    tensor = board_to_tensor(board).unsqueeze(0).to(device)  # (1, 768)

    with torch.no_grad():
        logits = model(tensor).squeeze(0)           # (4096,)

    # Mask illegal moves
    mask = get_legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")

    # Apply temperature and sample
    probs = F.softmax(logits / temperature, dim=0)
    best_idx = probs.argmax().item()

    move = index_to_move(best_idx, board)

    # Safety fallback: if somehow still illegal, pick first legal
    if move not in board.legal_moves:
        move = next(iter(board.legal_moves))

    # predict_move doesn't evaluate position score deeply, return 0.0
    return move, 0.0


# ── Quick smoke test ───────────────────────────────────
if __name__ == "__main__":
    from src.model import load_model
    model = load_model(os.path.join(config.CHECKPOINT_DIR, "best.pt"))
    board = chess.Board()
    for _ in range(5):
        move = predict_move(board, model)
        print("Predicted:", move, "(" + board.san(move) + ")")
        board.push(move)
