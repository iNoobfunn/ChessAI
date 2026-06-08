import os, sys
import chess, chess.pgn, csv, time
from tqdm import tqdm

# Add parent dir so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.engine import ChessAIEngine

try:
    from stockfish import Stockfish
    HAS_STOCKFISH = True
except ImportError:
    HAS_STOCKFISH = False

def play_game(engine: ChessAIEngine, stockfish: 'Stockfish',
              ai_is_white: bool) -> str:
    """Play one game. Returns '1-0', '0-1', or '1/2-1/2'."""
    board = chess.Board()
    stockfish.set_position([])

    while not board.is_game_over(claim_draw=True):
        ai_turn = (board.turn == chess.WHITE) == ai_is_white

        if ai_turn:
            move = engine.get_move(board)
        else:
            stockfish.set_fen_position(board.fen())
            sf_move = stockfish.get_best_move_time(100)  # 100ms per move
            move = chess.Move.from_uci(sf_move)

        board.push(move)
        stockfish.make_moves_from_current_position([move.uci()])

    return board.result()


def run_tournament(checkpoint_path: str = None):
    if not HAS_STOCKFISH:
        print("ERROR: stockfish python package not installed. Run: pip install stockfish")
        return

    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best.pt")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    engine = ChessAIEngine(checkpoint_path)

    csv_path = os.path.join(config.RESULTS_DIR, "tournament.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["elo_level", "game_num", "ai_color", "result", "ai_won"])

        for elo in config.STOCKFISH_ELO_LEVELS:
            sf = Stockfish(path=config.STOCKFISH_PATH,
                           parameters={"UCI_LimitStrength": True,
                                        "UCI_Elo": elo})

            wins = draws = losses = 0
            print(f"\n── vs Stockfish Elo {elo} ──────────────────")

            for g in tqdm(range(config.TOURNAMENT_GAMES), desc=f"Elo {elo}"):
                ai_white = (g % 2 == 0)
                result = play_game(engine, sf, ai_white)

                ai_won = ("1-0" == result and ai_white) or \
                         ("0-1" == result and not ai_white)
                draw   = result == "1/2-1/2"
                ai_lost = not ai_won and not draw

                wins   += int(ai_won)
                draws  += int(draw)
                losses += int(ai_lost)

                writer.writerow([elo, g, "W" if ai_white else "B",
                                 result, int(ai_won)])
                csvfile.flush()

            n = config.TOURNAMENT_GAMES
            print(f"Elo {elo}: W{wins} D{draws} L{losses} | "
                  f"Score {(wins + draws*0.5)/n*100:.1f}%")

    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    run_tournament()
