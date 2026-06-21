"""
Tiny built-in opening book (#7). Maps a position (by its board FEN, ignoring the
move-clock fields) to a list of solid candidate replies in UCI. When several
replies exist, one is chosen at random for variety. Returns None when the
position is not in the book, so the caller can fall back to the net / search.
"""
import random
import chess

# Keyed by the first 4 fields of the FEN (pieces, turn, castling, en-passant).
# Values are lists of reasonable continuations in UCI notation.
_BOOK_SAN = {
    # ── First move as White ──────────────────────────────
    "start": ["e2e4", "d2d4", "g1f3", "c2c4"],

    # ── Black replies to 1.e4 ────────────────────────────
    "1.e4":  ["e7e5", "c7c5", "e7e6", "c7c6"],          # open / Sicilian / French / Caro
    # ── Black replies to 1.d4 ────────────────────────────
    "1.d4":  ["d7d5", "g8f6", "e7e6"],
    # ── Black replies to 1.Nf3 / 1.c4 ────────────────────
    "1.Nf3": ["d7d5", "g8f6", "c7c5"],
    "1.c4":  ["e7e5", "g8f6", "c7c5"],

    # ── White's 2nd move in the open game (1.e4 e5) ──────
    "1.e4 e5": ["g1f3", "f1c4", "b1c3"],                # Italian / Spanish setups
    # ── Sicilian (1.e4 c5) ───────────────────────────────
    "1.e4 c5": ["g1f3", "b1c3", "c2c3"],
    # ── 1.d4 d5 ──────────────────────────────────────────
    "1.d4 d5": ["c2c4", "g1f3"],                        # Queen's Gambit
    # ── 1.d4 Nf6 ─────────────────────────────────────────
    "1.d4 Nf6": ["c2c4", "g1f3"],
}


def _key(board: chess.Board):
    """Map the current board to a book key based on the move sequence so far."""
    moves = board.move_stack
    if len(moves) == 0:
        return "start"
    # Reconstruct a short SAN-ish signature by replaying.
    tmp = chess.Board()
    sans = []
    for m in moves:
        sans.append(tmp.san(m))
        tmp.push(m)
    if len(sans) == 1:
        return f"1.{sans[0]}"
    if len(sans) == 2:
        return f"1.{sans[0]} {sans[1]}"
    return None


def book_move(board: chess.Board):
    """Return a legal chess.Move from the book, or None if out of book."""
    key = _key(board)
    if key is None:
        return None
    candidates = _BOOK_SAN.get(key)
    if not candidates:
        return None
    legal = []
    for uci in candidates:
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            continue
        if mv in board.legal_moves:
            legal.append(mv)
    if not legal:
        return None
    return random.choice(legal)
