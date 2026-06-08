import os, sys
import chess

# Add parent dir so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.model import load_model
from src.minimax import search

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

    def get_move(self, board: chess.Board,
                   use_search: bool = True) -> chess.Move:
        """
        Returns the engine's chosen move.
        use_search=True : minimax depth-3 (recommended)
        use_search=False: pure neural net prediction (faster, weaker)
        """
        if use_search:
            return search(board, self.model, self.device)
        else:
            from src.inference import predict_move
            return predict_move(board, self.model, self.device)


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
