"""
Estimate the bot's Elo by a gauntlet against Stockfish at calibrated UCI_Elo
levels, then converting the score into a performance rating.

  performance = R_opp + 400 * log10(S / (1 - S))

S=0 or S=1 are clamped to ±0.5 of a game so the log stays finite. We report a
per-level performance rating plus an overall estimate (pooled across games,
weighted toward the level whose score is nearest 50% — the most informative one).

Run:  python estimate_elo.py
"""
import os, sys, math, time, argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import config
from stockfish import Stockfish
import chess

# ── Settings (a fixed, fast time control for the measurement) ──
AI_THINK_SEC   = 0.40
SF_MOVETIME_MS = 40

# (UCI_Elo level, number of games)
LADDER = [(1320, 20), (1500, 10), (1700, 6)]


def perf_rating(opp_elo, score, n):
    """Performance rating from a score fraction over n games (clamped)."""
    s = min(max(score, 0.5 / n), 1 - 0.5 / n)
    return opp_elo + 400 * math.log10(s / (1 - s))


def elo_margin(score, n):
    """Rough ±Elo from the binomial standard error of the score."""
    s = min(max(score, 0.5 / n), 1 - 0.5 / n)
    se = math.sqrt(s * (1 - s) / n)
    # derivative of perf wrt score = 400 / (ln10 * s(1-s))
    return 400 / (math.log(10) * s * (1 - s)) * se


def play_game(engine, sf, ai_white):
    board = chess.Board()
    while not board.is_game_over(claim_draw=True):
        ai_turn = (board.turn == chess.WHITE) == ai_white
        if ai_turn:
            move, _ = engine.get_move_with_eval(board, use_book=True)
        else:
            sf.set_fen_position(board.fen())
            move = chess.Move.from_uci(sf.get_best_move_time(SF_MOVETIME_MS))
        board.push(move)
    return board.result(claim_draw=True)


def main():
    config.TIME_LIMIT_SEC = AI_THINK_SEC
    from src.engine import ChessAIEngine
    engine = ChessAIEngine()

    print(f"\nElo gauntlet | AI think {AI_THINK_SEC}s/move vs Stockfish {SF_MOVETIME_MS}ms/move")
    print(f"Checkpoint: {os.path.join(config.CHECKPOINT_DIR, 'best.pt')}\n")

    results = []
    total_t0 = time.time()
    for elo, n in LADDER:
        sf = Stockfish(path=config.STOCKFISH_PATH,
                       parameters={"UCI_LimitStrength": True, "UCI_Elo": elo})
        w = d = l = 0
        for g in range(n):
            ai_white = (g % 2 == 0)
            t0 = time.time()
            res = play_game(engine, sf, ai_white)
            ai_won = (res == "1-0" and ai_white) or (res == "0-1" and not ai_white)
            draw = res == "1/2-1/2"
            if ai_won: w += 1
            elif draw: d += 1
            else: l += 1
            tag = "win " if ai_won else ("draw" if draw else "loss")
            print(f"  SF{elo} g{g+1:>2}/{n} {'W' if ai_white else 'B'} -> {res} ({tag})  "
                  f"[{time.time()-t0:4.1f}s]  running W{w} D{d} L{l}")

        score = (w + 0.5 * d) / n
        pr = perf_rating(elo, score, n)
        mg = elo_margin(score, n)
        results.append((elo, n, w, d, l, score, pr, mg))
        print(f"  => vs SF {elo}: {w}W {d}D {l}L  score {score*100:4.1f}%  "
              f"perf ~{pr:.0f} (±{mg:.0f})\n")

    # ── Summary ───────────────────────────────────────────
    print("=" * 64)
    print(f"{'SF Elo':>7} {'G':>3} {'W':>3} {'D':>3} {'L':>3} {'Score':>7} {'Perf':>7} {'±':>5}")
    for elo, n, w, d, l, score, pr, mg in results:
        print(f"{elo:>7} {n:>3} {w:>3} {d:>3} {l:>3} {score*100:>6.1f}% {pr:>7.0f} {mg:>5.0f}")

    # Overall: weight each level's performance rating by informativeness
    # (closeness of its score to 50%) and by games played.
    wsum = psum = 0.0
    for elo, n, w, d, l, score, pr, mg in results:
        weight = n * (1 - abs(score - 0.5) * 2) ** 2 + 1e-6
        wsum += weight
        psum += weight * pr
    overall = psum / wsum
    games_total = sum(r[1] for r in results)
    print("-" * 64)
    print(f"Overall estimated Elo: ~{overall:.0f}  "
          f"(Stockfish UCI_Elo scale, {games_total} games, "
          f"{time.time()-total_t0:.0f}s)")
    print("Note: this scale runs ~strong; Lichess blitz ratings tend to be higher.")


if __name__ == "__main__":
    main()
