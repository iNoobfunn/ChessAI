import os, sys
import chess

# Add parent dir so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.model import load_model
from src.minimax import search
from src.inference import predict_move
from src.opening_book import book_move


class ChessAIEngine:
    """
    Main interface for the chess AI.
    Usage:
        engine = ChessAIEngine("checkpoints/best.pt")
        move = engine.get_move(board)
    """
    def __init__(self, checkpoint_path: str = None, device: str = config.DEVICE):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best.pt")
        self.device = device
        self.model  = load_model(checkpoint_path, device)
        print(f"Engine loaded from {checkpoint_path} on {device}")

    def get_move_with_eval(self, board: chess.Board, use_search: bool = True,
                           use_book: bool = True):
        """Return (move, eval_centipawns)."""
        # Opening book / sampled variety for the first few plies (#7)
        if use_book and len(board.move_stack) < config.OPENING_BOOK_PLIES:
            bm = book_move(board)
            if bm is not None:
                return bm, 0.0
            move, _ = predict_move(board, self.model, self.device,
                                   temperature=config.OPENING_TEMPERATURE)
            return move, 0.0

        if use_search:
            return search(board, self.model, self.device,
                          depth=config.SEARCH_DEPTH,
                          time_limit=config.TIME_LIMIT_SEC)
        move, val = predict_move(board, self.model, self.device, temperature=0.0)
        return move, val * 1000.0

    def get_move(self, board: chess.Board, use_search: bool = True) -> chess.Move:
        """Return just the chosen move."""
        move, _ = self.get_move_with_eval(board, use_search=use_search)
        return move


# ── Play a quick game vs random mover to verify ────────
if __name__ == "__main__":
    import random
    engine = ChessAIEngine()
    board  = chess.Board()
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = engine.get_move(board)
        else:
            move = random.choice(list(board.legal_moves))
        board.push(move)
    print("Result:", board.result())
    print(board)
