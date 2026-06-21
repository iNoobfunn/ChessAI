import bz2, chess, chess.pgn, io, os, sys
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Add parent dir so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.board_encoder import board_to_planes, move_to_index, N_PLANES
import config

# ── PGN streaming parser ───────────────────────────────

def _open_pgn(pgn_path: str):
    if pgn_path.endswith(".zst"):
        import zstandard as zstd
        fh = open(pgn_path, "rb")
        dctx = zstd.ZstdDecompressor()
        return io.TextIOWrapper(dctx.stream_reader(fh), encoding="utf-8", errors="ignore")
    elif pgn_path.endswith(".bz2"):
        return bz2.open(pgn_path, "rt", encoding="utf-8", errors="ignore")
    return open(pgn_path, "rt", encoding="utf-8", errors="ignore")


def build_dataset(pgn_path: str, out_path: str):
    """
    Stream a (optionally compressed) PGN file, filter by Elo, and save arrays:
        X  (N, 18, 8, 8) float32  board planes
        y  (N,)          int64    target move index (winner's move)
        v  (N,)          float32  value target in {-1, 0, +1} from side-to-move POV
        pw (N,)          float32  policy weight: 1.0 if side-to-move won, else 0.0

    Policy loss is applied only where pw == 1 (learn from the winner's moves, #5),
    while the value head learns from every position including draws (#1).
    """
    X_list, y_list, v_list, pw_list = [], [], [], []
    games_kept = 0

    f = _open_pgn(pgn_path)
    with f:
        pbar = tqdm(desc="Parsing games", unit="game")
        while games_kept < config.MAX_GAMES:
            try:
                game = chess.pgn.read_game(f)
            except Exception:
                continue
            if game is None:
                break

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

            result = headers.get("Result", "*")
            if result == "1-0":
                winner = chess.WHITE
            elif result == "0-1":
                winner = chess.BLACK
            elif result == "1/2-1/2" and config.INCLUDE_DRAWS:
                winner = None
            else:
                continue

            board = game.board()
            move_count = 0
            for move in game.mainline_moves():
                if move_count >= config.MAX_MOVES_PER_GAME:
                    break
                if not board.is_legal(move):
                    break

                stm = board.turn
                if winner is None:
                    value, pw = 0.0, 0.0
                elif winner == stm:
                    value, pw = 1.0, 1.0       # winner to move → imitate this move
                else:
                    value, pw = -1.0, 0.0      # losing side → value signal only

                X_list.append(board_to_planes(board))
                y_list.append(move_to_index(move, board))
                v_list.append(value)
                pw_list.append(pw)

                board.push(move)
                move_count += 1

            games_kept += 1
            pbar.update(1)
        pbar.close()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    X  = np.stack(X_list).astype(np.float32)
    y  = np.array(y_list,  dtype=np.int64)
    v  = np.array(v_list,  dtype=np.float32)
    pw = np.array(pw_list, dtype=np.float32)
    np.savez_compressed(out_path, X=X, y=y, v=v, pw=pw)
    print(f"Saved {len(X)} positions from {games_kept} games "
          f"({int(pw.sum())} winner-moves for policy) -> {out_path}")
    return X, y, v, pw


# ── PyTorch Dataset ───────────────────────────────────

class ChessDataset(Dataset):
    def __init__(self, npz_path: str):
        import torch
        data = np.load(npz_path)
        # Store planes as float16 to halve host RAM (~5GB instead of ~10GB);
        # cast back to float32 per sample in __getitem__.
        self.X  = torch.from_numpy(data["X"]).to(torch.float16)
        self.y  = torch.tensor(data["y"],  dtype=torch.long)
        self.v  = torch.tensor(data["v"],  dtype=torch.float32)
        self.pw = torch.tensor(data["pw"], dtype=torch.float32)

    def __len__(self):    return len(self.X)
    def __getitem__(self, i):
        return self.X[i].float(), self.y[i], self.v[i], self.pw[i]


def get_dataloaders(npz_path: str):
    import torch
    ds = ChessDataset(npz_path)
    n_val = int(len(ds) * config.VAL_SPLIT)
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader


# Run this file directly to build the dataset:
#   python src/dataset.py data/raw/lichess_2013-01.pgn.zst
if __name__ == "__main__":
    build_dataset(sys.argv[1], os.path.join(config.DATA_PROC_DIR, "dataset.npz"))
