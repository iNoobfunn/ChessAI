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
MAX_GAMES      = 60_000     # cap to keep dataset manageable
MAX_MOVES_PER_GAME = 80     # skip moves after this (endgame noise)
VAL_SPLIT      = 0.1
INCLUDE_DRAWS  = True       # keep draws (value head learns from 0-outcome positions)

# Model — residual CNN with policy + value heads
N_PLANES       = 18         # 12 piece + 4 castling + 1 en-passant + 1 fifty-move
INPUT_DIM      = N_PLANES * 64
OUTPUT_DIM     = 4096       # 64×64 from-to move encoding (promotions decode to queen)
CONV_CHANNELS  = 128        # channels in the residual tower
N_RES_BLOCKS   = 6          # depth of the residual tower
DROPOUT        = 0.3
VALUE_LOSS_WEIGHT = 1.0     # weight of the value-head MSE term in the combined loss

# Training
BATCH_SIZE     = 1024
EPOCHS         = 18
LR             = 1e-3
LR_STEP        = 4          # reduce LR every N epochs
LR_GAMMA       = 0.5
DEVICE         = "cuda"      # change to "cpu" if you don't have a GPU

# Minimax
SEARCH_DEPTH   = 6          # max depth for iterative deepening
TIME_LIMIT_SEC = 2.0        # max seconds per move
USE_NN_EVAL    = True       # blend NN value head into the leaf evaluation
NN_EVAL_WEIGHT = 0.6        # 0 = pure handcrafted PST, 1 = pure NN value head
# The NN value call costs ~3ms (unbatched GPU), so only blend it at nodes within
# this many plies of the root; deeper leaves use the fast handcrafted eval. This
# lets iterative deepening reach much greater depth while keeping NN judgment up top.
NN_EVAL_MAX_PLY = 1

# Opening book / variety
OPENING_BOOK_PLIES = 8      # use book / sampling for the first N plies
OPENING_TEMPERATURE = 0.85  # softmax temperature for opening move sampling

# Tournament
TOURNAMENT_GAMES    = 4     # games per Elo level (each side twice)
TOURNAMENT_THINK_SEC = 0.6  # AI think time per move during the tournament
STOCKFISH_MOVETIME_MS = 100 # Stockfish think time per move
# NB: Stockfish 18's UCI_Elo floor is 1320 — lower values are clamped.
STOCKFISH_ELO_LEVELS = [1320, 1500, 1700]
