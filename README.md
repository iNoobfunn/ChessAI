# Chess AI

A supervised chess move predictor with minimax alpha-beta search wrapper.
Neural network (MLP) trained on Lichess games, with a beautiful web UI to play against it.

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
