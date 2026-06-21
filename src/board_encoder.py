import chess
import numpy as np
import torch

# Ordered list: (piece_type, color) pairs — defines plane index 0..11
PLANE_ORDER = [
    (chess.PAWN,   chess.WHITE), (chess.KNIGHT, chess.WHITE),
    (chess.BISHOP, chess.WHITE), (chess.ROOK,   chess.WHITE),
    (chess.QUEEN,  chess.WHITE), (chess.KING,   chess.WHITE),
    (chess.PAWN,   chess.BLACK), (chess.KNIGHT, chess.BLACK),
    (chess.BISHOP, chess.BLACK), (chess.ROOK,   chess.BLACK),
    (chess.QUEEN,  chess.BLACK), (chess.KING,   chess.BLACK),
]

# Plane layout (18 total):
#   0..11  piece planes (see PLANE_ORDER)
#   12     side-to-move (always-white) kingside castling rights  (full plane)
#   13     side-to-move queenside castling rights
#   14     opponent kingside castling rights
#   15     opponent queenside castling rights
#   16     en-passant target square (single square set to 1)
#   17     fifty-move-rule progress (halfmove_clock / 100, full plane)
N_PLANES = 18


def board_to_planes(board: chess.Board) -> np.ndarray:
    """
    Convert a chess.Board to an (18, 8, 8) float32 array.
    Always encodes from the side-to-move's perspective (mirror if Black to move),
    so the network only ever sees "white to move" positions.
    """
    if board.turn == chess.BLACK:
        board = board.mirror()   # flip so side-to-move is always "white"

    planes = np.zeros((N_PLANES, 8, 8), dtype=np.float32)

    for plane_idx, (piece_type, color) in enumerate(PLANE_ORDER):
        for sq in board.pieces(piece_type, color):
            r, f = divmod(sq, 8)
            planes[plane_idx, r, f] = 1.0

    # Castling rights (after the mirror, side-to-move is always White)
    if board.has_kingside_castling_rights(chess.WHITE):  planes[12, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE): planes[13, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):  planes[14, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK): planes[15, :, :] = 1.0

    # En-passant target square
    if board.ep_square is not None:
        r, f = divmod(board.ep_square, 8)
        planes[16, r, f] = 1.0

    # Fifty-move-rule progress, normalised to [0, 1]
    planes[17, :, :] = min(board.halfmove_clock, 100) / 100.0

    return planes


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """(18, 8, 8) float32 tensor for a single position (no batch dim)."""
    return torch.from_numpy(board_to_planes(board))


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """
    Encode a chess.Move as integer index in [0, 4095] = from_sq * 64 + to_sq.
    Mirrors the move when Black is to move so it matches board_to_tensor.
    Note: promotion piece is folded away here and recovered (as queen) on decode.
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
    Decode integer index back to a legal chess.Move for the given board.
    Handles board mirroring for Black, and re-attaches a queen promotion when a
    pawn lands on the last rank (fixes the previous illegal-move bug). Underpromotions
    are intentionally folded to queen promotion (>99.9% of human promotions are queens).
    """
    from_sq = idx // 64
    to_sq   = idx % 64
    if board.turn == chess.BLACK:
        from_sq = chess.square_mirror(from_sq)
        to_sq   = chess.square_mirror(to_sq)

    promotion = None
    piece = board.piece_at(from_sq)
    if piece is not None and piece.piece_type == chess.PAWN and \
       chess.square_rank(to_sq) in (0, 7):
        promotion = chess.QUEEN

    return chess.Move(from_sq, to_sq, promotion)


# ── Quick sanity test ──────────────────────────────────
# Run: python src/board_encoder.py
if __name__ == "__main__":
    b = chess.Board()
    t = board_to_tensor(b)
    assert t.shape == (N_PLANES, 8, 8), f"Wrong shape {t.shape}"
    assert t[:12].sum() == 32, "Should be 32 pieces in start pos"
    # castling planes all set in the start position
    assert t[12].sum() == 64 and t[15].sum() == 64, "Castling planes wrong"
    print("board_encoder OK — tensor shape:", tuple(t.shape))
