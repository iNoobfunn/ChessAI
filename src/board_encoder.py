import chess
import numpy as np
import torch

# Ordered list: (piece_type, color) pairs — defines plane index
PLANE_ORDER = [
    (chess.PAWN,   chess.WHITE), (chess.KNIGHT, chess.WHITE),
    (chess.BISHOP, chess.WHITE), (chess.ROOK,   chess.WHITE),
    (chess.QUEEN,  chess.WHITE), (chess.KING,   chess.WHITE),
    (chess.PAWN,   chess.BLACK), (chess.KNIGHT, chess.BLACK),
    (chess.BISHOP, chess.BLACK), (chess.ROOK,   chess.BLACK),
    (chess.QUEEN,  chess.BLACK), (chess.KING,   chess.BLACK),
]

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Convert a chess.Board to a flat (768,) float32 tensor.
    Always encodes from White's perspective (flip if Black to move).
    """
    if board.turn == chess.BLACK:
        board = board.mirror()   # flip so side-to-move is always "white"

    planes = np.zeros(12 * 64, dtype=np.float32)
    for plane_idx, (piece_type, color) in enumerate(PLANE_ORDER):
        for sq in board.pieces(piece_type, color):
            planes[plane_idx * 64 + sq] = 1.0

    return torch.tensor(planes, dtype=torch.float32)


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """
    Encode a chess.Move as integer index in [0, 4095].
    If board is Black to move, mirror the move to match board_to_tensor.
    Formula: from_sq * 64 + to_sq
    """
    if board.turn == chess.BLACK:
        move = chess.Move(
            chess.square_mirror(move.from_square),
            chess.square_mirror(move.to_square),
            move.promotion
        )
    return move.from_square * 64 + move.to_square


def index_to_move(idx: int, board: chess.Board) -> chess.Move:
    """
    Decode integer index back to chess.Move for the given board.
    Handles board mirroring for Black to move.
    """
    from_sq = idx // 64
    to_sq   = idx % 64
    if board.turn == chess.BLACK:
        from_sq = chess.square_mirror(from_sq)
        to_sq   = chess.square_mirror(to_sq)
    return chess.Move(from_sq, to_sq)


# ── Quick sanity test ──────────────────────────────────
# Run: python src/board_encoder.py
if __name__ == "__main__":
    b = chess.Board()
    t = board_to_tensor(b)
    assert t.shape == (768,), "Wrong shape"
    assert t.sum() == 32,     "Should be 32 pieces in start pos"
    print("board_encoder OK — tensor shape:", t.shape)
