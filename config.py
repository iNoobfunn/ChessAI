# ── config.py ─────────────────────────────────────────
# All project-wide settings. Edit this file only.

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STOCKFISH_PATH = os.path.join(BASE_DIR, "stockfish", "stockfish-windows-x86-64-avx2.exe")  # update to your path
DATA_RAW_DIR   = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")

# Dataset
MIN_ELO        = 1350       # filter: both players ≥ this
MAX_ELO        = 1650
MAX_GAMES      = 30_000     # cap to keep dataset manageable
MAX_MOVES_PER_GAME = 80     # skip moves after this (endgame noise)
VAL_SPLIT      = 0.1

# Model
INPUT_DIM      = 768        # 12 planes × 64 squares
HIDDEN_DIMS    = [1024, 512, 256]
OUTPUT_DIM     = 4096       # 64×64 from-to move encoding
DROPOUT        = 0.3

# Training
BATCH_SIZE     = 512
EPOCHS         = 50
LR             = 1e-3
LR_STEP        = 5          # reduce LR every N epochs
LR_GAMMA       = 0.5
DEVICE         = "cuda"      # change to "cpu" if you don't have a GPU

# Minimax
SEARCH_DEPTH   = 3          # depth 3 for faster responses
TIME_LIMIT_SEC = 2.0        # max seconds per move

# Tournament
TOURNAMENT_GAMES    = 50    # games per Elo level
STOCKFISH_ELO_LEVELS = [800, 1200, 1500]
