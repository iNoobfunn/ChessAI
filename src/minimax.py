import os, sys
import chess, chess.polyglot, time, torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.board_encoder import board_to_tensor, move_to_index
import config

PAWN_PST = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
]
KNIGHT_PST = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]
BISHOP_PST = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]
ROOK_PST = [
      0,  0,  0,  0,  0,  0,  0,  0,
      5, 10, 10, 10, 10, 10, 10,  5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      0,  0,  0,  5,  5,  0,  0,  0
]
QUEEN_PST = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]
KING_PST = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
]

PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
}
PSTS = {
    chess.PAWN: PAWN_PST, chess.KNIGHT: KNIGHT_PST, chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST, chess.QUEEN: QUEEN_PST, chess.KING: KING_PST
}

MATE = 99999.0

# Precompute combined (material value + piece-square) tables, indexed by square,
# one per (piece_type, color). For a white piece on `sq`, the PST is read from the
# vertically mirrored square (sq ^ 56); black reads `sq` directly. This lets
# evaluate_board iterate only the occupied squares with a single table lookup.
_VAL_PST = {}
for _pt in PIECE_VALUES:
    for _color in (chess.WHITE, chess.BLACK):
        _tbl = [0] * 64
        for _sq in range(64):
            _idx = (_sq ^ 56) if _color == chess.WHITE else _sq
            _tbl[_sq] = PIECE_VALUES[_pt] + PSTS[_pt][_idx]
        _VAL_PST[(_pt, _color)] = _tbl


def evaluate_board(board: chess.Board) -> float:
    """Handcrafted material + piece-square eval, from side-to-move perspective."""
    if board.is_checkmate():
        return -MATE
    if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
        return 0.0

    score = 0
    for sq, p in board.piece_map().items():
        t = _VAL_PST[(p.piece_type, p.color)][sq]
        score += t if p.color else -t          # WHITE is True, BLACK is False

    return score if board.turn else -score


class SearchTimeout(Exception):
    pass


# ── Per-search state (transposition table, killers, history, NN cache) ──
class _SearchCtx:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.tt = {}                 # zobrist -> (depth, flag, value, best_move)
        self.killers = {}            # ply -> [move, move]
        self.history = {}            # (from, to) -> score
        self.nn_cache = {}           # zobrist -> value scalar [-1, 1]
        self.nodes = 0

    def nn_value(self, board) -> float:
        if not config.USE_NN_EVAL or self.model is None:
            return 0.0
        key = chess.polyglot.zobrist_hash(board)
        cached = self.nn_cache.get(key)
        if cached is not None:
            return cached
        tensor = board_to_tensor(board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, value = self.model(tensor)
        v = float(value.item())
        self.nn_cache[key] = v
        return v


def _leaf_eval(board, ctx, ply=0):
    """
    Blend handcrafted eval with the NN value head (both side-to-move POV).
    The (expensive) NN value is only consulted near the root (ply <= NN_EVAL_MAX_PLY);
    deeper leaves use the fast handcrafted eval so search can reach greater depth.
    """
    pst = evaluate_board(board)
    if config.USE_NN_EVAL and ply <= config.NN_EVAL_MAX_PLY:
        nn_cp = ctx.nn_value(board) * 1000.0     # map [-1,1] -> centipawns
        w = config.NN_EVAL_WEIGHT
        return (1 - w) * pst + w * nn_cp
    return pst


def _order_moves(board, ctx, ply, tt_move):
    """Order moves: TT move, MVV-LVA captures, killers, then history heuristic."""
    killers = ctx.killers.get(ply, [])

    def score(m):
        if tt_move is not None and m == tt_move:
            return 1_000_000
        if board.is_capture(m):
            victim = board.piece_at(m.to_square)
            attacker = board.piece_at(m.from_square)
            vv = PIECE_VALUES[victim.piece_type] if victim else 100  # en-passant
            av = PIECE_VALUES[attacker.piece_type] if attacker else 100
            return 100_000 + vv * 10 - av
        if m in killers:
            return 90_000
        return ctx.history.get((m.from_square, m.to_square), 0)

    moves = list(board.legal_moves)
    moves.sort(key=score, reverse=True)
    return moves


def _quiescence(board, alpha, beta, deadline, ctx, depth_limit=8):
    if time.time() > deadline:
        raise SearchTimeout()
    ctx.nodes += 1

    stand_pat = evaluate_board(board)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat
    if depth_limit == 0:
        return alpha

    captures = list(board.generate_legal_captures())
    captures.sort(key=lambda m: (PIECE_VALUES[board.piece_at(m.to_square).piece_type]
                                 if board.piece_at(m.to_square) else 100), reverse=True)
    for move in captures:
        board.push(move)
        try:
            sc = -_quiescence(board, -beta, -alpha, deadline, ctx, depth_limit - 1)
        except SearchTimeout:
            board.pop(); raise
        board.pop()
        if sc >= beta:
            return beta
        if sc > alpha:
            alpha = sc
    return alpha


def _alphabeta(board, depth, alpha, beta, deadline, ctx, ply=0):
    if time.time() > deadline:
        raise SearchTimeout()
    ctx.nodes += 1
    alpha_orig = alpha

    key = chess.polyglot.zobrist_hash(board)
    entry = ctx.tt.get(key)
    tt_move = None
    if entry is not None and entry[0] >= depth:
        e_depth, e_flag, e_val, e_move = entry
        if e_flag == 0:                      # EXACT
            return e_val, e_move
        elif e_flag == 1:                    # LOWER bound
            alpha = max(alpha, e_val)
        elif e_flag == 2:                    # UPPER bound
            beta = min(beta, e_val)
        if alpha >= beta:
            return e_val, e_move
    if entry is not None:
        tt_move = entry[3]

    if board.is_checkmate():
        return -MATE - depth, None          # prefer faster mates
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0.0, None
    if depth == 0:
        return _leaf_eval(board, ctx, ply), None

    best_move = None
    value = float("-inf")
    for move in _order_moves(board, ctx, ply, tt_move):
        board.push(move)
        try:
            child, _ = _alphabeta(board, depth - 1, -beta, -alpha, deadline, ctx, ply + 1)
        except SearchTimeout:
            board.pop(); raise
        child = -child
        board.pop()

        if child > value:
            value = child
            best_move = move
        alpha = max(alpha, value)
        if alpha >= beta:
            # Beta cutoff: record killer + history for quiet moves
            if not board.is_capture(move):
                kl = ctx.killers.setdefault(ply, [])
                if move not in kl:
                    kl.insert(0, move)
                    del kl[2:]
                k = (move.from_square, move.to_square)
                ctx.history[k] = ctx.history.get(k, 0) + depth * depth
            break

    # Store in transposition table
    flag = 0 if alpha_orig < value < beta else (1 if value >= beta else 2)
    ctx.tt[key] = (depth, flag, value, best_move)
    return value, best_move


def search(board: chess.Board, model, device=config.DEVICE,
           depth=config.SEARCH_DEPTH, time_limit=config.TIME_LIMIT_SEC):
    """
    Iterative-deepening negamax alpha-beta with NN value-head leaf eval,
    transposition table, killer/history move ordering, and NN policy ordering
    of root moves. Returns (best_move, value_in_centipawns).
    """
    ctx = _SearchCtx(model, device)
    deadline = time.time() + time_limit

    # NN policy for root move ordering
    root_moves = list(board.legal_moves)
    if model is not None:
        from src.inference import get_legal_move_mask
        tensor = board_to_tensor(board).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = model(tensor)
        logits = logits.squeeze(0)
        mask = get_legal_move_mask(board).to(device)
        logits[~mask] = float("-inf")
        probs = F.softmax(logits, dim=0)
        root_moves.sort(key=lambda m: probs[move_to_index(m, board)].item(), reverse=True)

    best_move = root_moves[0]
    best_value = 0.0

    # Iterative deepening
    for d in range(1, depth + 1):
        alpha, beta = float("-inf"), float("inf")
        value = float("-inf")
        local_best = best_move
        ordered = [best_move] + [m for m in root_moves if m != best_move]
        try:
            for move in ordered:
                board.push(move)
                child, _ = _alphabeta(board, d - 1, -beta, -alpha, deadline, ctx, 1)
                child = -child
                board.pop()
                if child > value:
                    value = child
                    local_best = move
                alpha = max(alpha, value)
        except SearchTimeout:
            if board.move_stack and board.peek() in root_moves:
                board.pop()
            break
        best_move, best_value = local_best, value
        if best_value > MATE / 2:            # found a forced mate; stop early
            break

    return best_move, best_value
