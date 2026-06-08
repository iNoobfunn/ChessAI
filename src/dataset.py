import bz2, chess, chess.pgn, io, os, sys
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Add parent dir so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.board_encoder import board_to_tensor, move_to_index
import config

# ── PGN streaming parser ───────────────────────────────

def build_dataset(pgn_path: str, out_path: str):
    """
    Stream a compressed PGN file (.bz2 or .zst), filter by Elo,
    and save (X, y) numpy arrays to out_path.
    """
    X_list, y_list = [], []
    games_kept = 0

    if pgn_path.endswith(".zst"):
        import zstandard as zstd
        fh = open(pgn_path, "rb")
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(fh)
        f = io.TextIOWrapper(stream_reader, encoding="utf-8", errors="ignore")
    elif pgn_path.endswith(".bz2"):
        f = bz2.open(pgn_path, "rt", encoding="utf-8", errors="ignore")
    else:
        f = open(pgn_path, "rt", encoding="utf-8", errors="ignore")

    with f:
        pbar = tqdm(desc="Parsing games", unit="game")
        while games_kept < config.MAX_GAMES:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            # Elo filter
            headers = game.headers
            try:
                w_elo = int(headers.get("WhiteElo", "0"))
                b_elo = int(headers.get("BlackElo", "0"))
            except ValueError:
                continue

            if not (config.MIN_ELO <= w_elo <= config.MAX_ELO):
                continue
            if not (config.MIN_ELO <= b_elo <= config.MAX_ELO):
                continue

            # Skip draws and incomplete games
            result = headers.get("Result", "*")
            if result not in ["1-0", "0-1"]:
                continue

            # Replay moves and collect (board, move) pairs
            board = game.board()
            move_count = 0
            for move in game.mainline_moves():
                if move_count >= config.MAX_MOVES_PER_GAME:
                    break
                if not board.is_legal(move):
                    break

                X_list.append(board_to_tensor(board).numpy())
                y_list.append(move_to_index(move, board))
                board.push(move)
                move_count += 1

            games_kept += 1
            pbar.update(1)

        pbar.close()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    np.savez_compressed(out_path, X=X, y=y)
    print(f"Saved {len(X)} positions from {games_kept} games -> {out_path}")
    return X, y


# ── PyTorch Dataset ───────────────────────────────────

class ChessDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        import torch
        self.X = torch.tensor(data["X"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.long)

    def __len__(self):    return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def get_dataloaders(npz_path: str):
    import torch

    ds = ChessDataset(npz_path)
    n_val = int(len(ds) * config.VAL_SPLIT)
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader


# Run this file directly to build the dataset:
# python src/dataset.py data/raw/lichess_db_standard_rated_2024-01.pgn.zst
if __name__ == "__main__":
    build_dataset(sys.argv[1], os.path.join(config.DATA_PROC_DIR, "dataset.npz"))
