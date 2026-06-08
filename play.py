"""
Interactive chess game vs the AI.
Run: python play.py
You play as White. Enter moves in UCI format e.g. e2e4
"""
import chess, sys, os

# Ensure we're in the right directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.engine import ChessAIEngine

engine = ChessAIEngine()
board  = chess.Board()

print("\nChess AI Demo — you are White. Enter moves as UCI (e.g. e2e4)\n")
print(board)
print()

while not board.is_game_over():
    if board.turn == chess.WHITE:
        while True:
            raw = input("Your move: ").strip()
            if raw.lower() in ('quit', 'exit', 'q'):
                print("Game abandoned.")
                sys.exit(0)
            try:
                move = chess.Move.from_uci(raw)
                if move in board.legal_moves:
                    break
                print("Illegal move. Try again.")
            except:
                print("Invalid format. Use e.g. e2e4")
    else:
        print("AI thinking...")
        move = engine.get_move(board)
        print(f"AI plays: {board.san(move)} ({move.uci()})")

    board.push(move)
    print("\n" + str(board) + "\n")

print("Game over:", board.result())
