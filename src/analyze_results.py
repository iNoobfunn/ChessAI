import os, sys
import pandas as pd

# Add parent dir so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def analyze():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    df = pd.read_csv(os.path.join(config.RESULTS_DIR, "tournament.csv"))

    # ── Summary table ─────────────────────────────────────
    summary = []
    for elo, grp in df.groupby("elo_level"):
        n     = len(grp)
        wins  = grp["ai_won"].sum()
        draws = (grp["result"] == "1/2-1/2").sum()
        losses = n - wins - draws
        score  = (wins + draws * 0.5) / n * 100
        summary.append({"Stockfish Elo": elo, "Games": n,
                         "Wins": wins, "Draws": draws,
                         "Losses": losses, "Score %": round(score, 1)})

    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(config.RESULTS_DIR, "summary.csv"), index=False)

    # ── Bar chart: score % per Elo level ──────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4ade80" if s >= 50 else "#f87171"
              for s in summary_df["Score %"]]
    ax.bar(summary_df["Stockfish Elo"].astype(str),
           summary_df["Score %"], color=colors, width=0.5)
    ax.axhline(50, color="white", lw=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Stockfish Elo level"); ax.set_ylabel("AI Score %")
    ax.set_title("Chess AI performance vs Stockfish")
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "score_by_elo.png"), dpi=150)
    plt.close()
    print("Chart saved → results/score_by_elo.png")

if __name__ == "__main__":
    analyze()
