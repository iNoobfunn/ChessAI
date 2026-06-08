import os, sys
import chess, time, torch
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

def evaluate_board(board: chess.Board) -> float:
    if board.is_checkmate():
        return -99999.0
    if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
        return 0.0

    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    psts = {
        chess.PAWN: PAWN_PST,
        chess.KNIGHT: KNIGHT_PST,
        chess.BISHOP: BISHOP_PST,
        chess.ROOK: ROOK_PST,
        chess.QUEEN: QUEEN_PST,
        chess.KING: KING_PST
    }
    
    score = 0.0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            val = piece_values[p.piece_type]
            pst = psts[p.piece_type]
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            
            if p.color == chess.WHITE:
                pst_idx = (7 - rank) * 8 + file
                score += val + pst[pst_idx]
            else:
                pst_idx = rank * 8 + file
                score -= val + pst[pst_idx]
                
    return score if board.turn == chess.WHITE else -score

class SearchTimeout(Exception):
    pass

def _quiescence(board, alpha, beta, deadline, depth_limit=10):
    if time.time() > deadline:
        raise SearchTimeout()

    stand_pat = evaluate_board(board)
    
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat
        
    if depth_limit == 0:
        return alpha
        
    captures = list(board.generate_legal_captures())
    
    for move in captures:
        board.push(move)
        try:
            score = -_quiescence(board, -beta, -alpha, deadline, depth_limit - 1)
        except SearchTimeout:
            board.pop()
            raise
        board.pop()
        
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
            
    return alpha

def _alphabeta(board, depth, alpha, beta, deadline):
    if time.time() > deadline:
        raise SearchTimeout()

    if board.is_checkmate():
        return -99999.0, None
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0.0, None

    if depth == 0:
        return _quiescence(board, alpha, beta, deadline), None

    best_move = None
    value = float("-inf")
    
    moves = list(board.legal_moves)
    # Simple move ordering: captures first
    moves.sort(key=lambda m: board.is_capture(m), reverse=True)
    
    for move in moves:
        board.push(move)
        try:
            child_val, _ = _alphabeta(board, depth - 1, -beta, -alpha, deadline)
        except SearchTimeout:
            board.pop()
            raise
        child_val = -child_val
        board.pop()
        
        if child_val > value:
            value = child_val
            best_move = move
            
        alpha = max(alpha, value)
        if alpha >= beta:
            break

    return value, best_move

def search(board: chess.Board, model, device=config.DEVICE,
            depth=config.SEARCH_DEPTH, time_limit=config.TIME_LIMIT_SEC) -> chess.Move:
            
    # 1. Ask Neural Network for root move probabilities
    from src.inference import get_legal_move_mask
    tensor = board_to_tensor(board).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor).squeeze(0)
    
    mask = get_legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")
    probs = F.softmax(logits, dim=0)
    
    legal_moves = list(board.legal_moves)
    move_probs = []
    for m in legal_moves:
        idx = move_to_index(m, board)
        move_probs.append((m, probs[idx].item()))
        
    # Sort root moves by NN probability to get perfect move ordering
    move_probs.sort(key=lambda x: x[1], reverse=True)
    
    deadline = time.time() + time_limit
    best_move = move_probs[0][0] # Fallback to NN best move if time runs out instantly
    
    # 2. Run Negamax Alpha-Beta search on the ordered moves
    alpha = float("-inf")
    beta = float("inf")
    value = float("-inf")
    
    for move, prob in move_probs:
        if time.time() > deadline: break
        
        board.push(move)
        try:
            child_val, _ = _alphabeta(board, depth - 1, -beta, -alpha, deadline)
        except SearchTimeout:
            board.pop()
            break
            
        child_val = -child_val
        board.pop()
        
        if child_val > value:
            value = child_val
            best_move = move
            
        alpha = max(alpha, value)
        
    return best_move, value
