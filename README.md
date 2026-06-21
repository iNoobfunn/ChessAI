# Chess AI

A supervised chess engine: a **residual CNN with policy + value heads** trained on
Lichess games, wrapped in an **iterative-deepening alpha-beta search** that uses the
value head as its leaf evaluator. Comes with a web UI to play against it.

## Architecture highlights

- **Input:** 18 board planes (12 pieces + castling rights + en-passant + fifty-move
  clock), always from the side-to-move's perspective.
- **Network:** residual conv tower (`board_encoder` → stem → N res-blocks) with two heads
  - *policy* → 4096 from-to move logits (illegal moves masked; promotions decode to queen)
  - *value* → scalar in [-1, 1], expected outcome for the side to move
- **Training:** policy cross-entropy is learned only from the **winning side's moves**;
  the value head learns from **every** position (including draws).
- **Search:** iterative deepening + negamax alpha-beta, transposition table (Zobrist),
  killer-move & history heuristics, MVV-LVA capture ordering, NN policy root ordering,
  quiescence search, and a blended handcrafted-PST / NN-value leaf evaluation.
- **Play:** small opening book + temperature sampling for opening variety; per-session
  games in the web server.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web UI
python app.py
# Open http://localhost:5000
```

## Project Structure

```
chess_ai/
├── src/
│   ├── board_encoder.py   # Board → 768-dim tensor
│   ├── dataset.py         # PGN parser + Dataset class
│   ├── model.py           # MLP definition
│   ├── train.py           # Training loop
│   ├── inference.py       # Move prediction at runtime
│   ├── minimax.py         # Alpha-beta search wrapper
│   ├── engine.py          # Main engine class
│   ├── tournament.py      # vs Stockfish evaluation
│   └── analyze_results.py # Results analysis + plots
├── web/
│   ├── index.html         # Chess UI
│   └── static/
│       ├── style.css      # Premium dark theme
│       └── app.js         # Interactive board controller
├── data/                  # Raw PGN and processed tensors
├── checkpoints/           # Saved model weights
├── results/               # Tournament logs, charts
├── config.py              # All hyperparameters
├── app.py                 # Flask web server
├── play.py                # Terminal demo
└── requirements.txt
```

## Training (Optional)

1. Download a Lichess PGN dump from database.lichess.org
2. Place the .pgn.zst file in `data/raw/`
3. Run: `python src/dataset.py data/raw/<file>.pgn.zst`
4. Run: `python src/train.py`

The web UI works without training (uses random weights as fallback).
