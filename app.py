"""
Flask web server for Chess AI — play against the bot in a browser.
Run: python app.py
Then open http://localhost:5000
"""
import os, sys, math, uuid

# Ensure chess_ai is the working context
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import chess

import config
from src.engine import ChessAIEngine

app = Flask(__name__, template_folder="web", static_folder="web/static")
app.secret_key = os.environ.get("CHESS_AI_SECRET", "chess-ai-dev-secret")
CORS(app, supports_credentials=True)

# ── Load the AI engine once ─────────────────────────────
engine = ChessAIEngine()
print(f"[OK] Chess AI engine ready on {config.DEVICE}")

# ── Per-session game state (fixes the old shared-global board, #8) ──
GAMES: dict[str, chess.Board] = {}


def _board() -> chess.Board:
    """Return (creating if needed) the board for the current browser session."""
    sid = session.get("sid")
    if sid is None or sid not in GAMES:
        sid = uuid.uuid4().hex
        session["sid"] = sid
        GAMES[sid] = chess.Board()
    return GAMES[sid]


def _set_board(board: chess.Board):
    GAMES[session["sid"]] = board


def sanitize_score(score):
    if math.isinf(score):
        return 999.99 if score > 0 else -999.99
    if math.isnan(score):
        return 0.0
    return round(score / 100.0, 2)


def _get_ai_move(board):
    """AI move via book → search, with a safe fallback to direct prediction."""
    try:
        return engine.get_move_with_eval(board)
    except Exception as e:
        print(f"Search failed: {e}, falling back to direct prediction")
        from src.inference import predict_move
        mv, val = predict_move(board, engine.model, engine.device, temperature=0.0)
        return mv, val * 1000.0


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/new_game", methods=["POST"])
def new_game():
    data = request.get_json() or {}
    player_color = data.get("player_color", "white")
    _board()                     # ensure a session id exists
    board = chess.Board()
    _set_board(board)

    response = {
        "fen": board.fen(),
        "player_color": player_color,
        "game_over": False,
        "result": None,
        "ai_move": None,
    }

    # If player chose black, AI makes the first move
    if player_color == "black":
        ai_move, raw_score = _get_ai_move(board)
        board.push(ai_move)
        response["fen"] = board.fen()
        response["ai_move"] = ai_move.uci()
        response["eval_score"] = sanitize_score(raw_score)

    return jsonify(response)


@app.route("/api/move", methods=["POST"])
def make_move():
    board = _board()
    data = request.get_json()
    uci_move = data.get("move")

    try:
        move = chess.Move.from_uci(uci_move)
    except Exception:
        return jsonify({"error": "Invalid move format"}), 400

    # Handle promotion: if a pawn reaches the last rank, default to queen
    piece = board.piece_at(move.from_square)
    if piece and piece.piece_type == chess.PAWN:
        if chess.square_rank(move.to_square) in (0, 7) and move.promotion is None:
            move = chess.Move(move.from_square, move.to_square, chess.QUEEN)

    if move not in board.legal_moves:
        return jsonify({"error": "Illegal move",
                        "legal_moves": [m.uci() for m in board.legal_moves]}), 400

    san = board.san(move)
    board.push(move)
    player_fen = board.fen()

    response = {
        "fen": board.fen(),
        "player_fen": player_fen,
        "player_move_san": san,
        "game_over": board.is_game_over(),
        "result": _get_result(board),
        "ai_move": None,
        "ai_move_san": None,
    }

    if not board.is_game_over():
        try:
            ai_move, raw_score = _get_ai_move(board)
            ai_san = board.san(ai_move)
            ai_is_white = board.turn == chess.WHITE
            std_score = raw_score if ai_is_white else -raw_score

            board.push(ai_move)
            response["fen"] = board.fen()
            response["ai_move"] = ai_move.uci()
            response["ai_move_san"] = ai_san
            response["eval_score"] = sanitize_score(std_score)
            response["game_over"] = board.is_game_over()
            response["result"] = _get_result(board)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"AI crashed: {str(e)}"}), 500

    return jsonify(response)


@app.route("/api/legal_moves", methods=["GET"])
def legal_moves():
    board = _board()
    moves = [{"uci": m.uci(),
              "from": chess.square_name(m.from_square),
              "to": chess.square_name(m.to_square)}
             for m in board.legal_moves]
    return jsonify({"moves": moves, "fen": board.fen()})


@app.route("/api/undo", methods=["POST"])
def undo():
    board = _board()
    if len(board.move_stack) >= 2:
        board.pop()  # undo AI
        board.pop()  # undo player
    elif len(board.move_stack) == 1:
        board.pop()
    return jsonify({"fen": board.fen(), "game_over": board.is_game_over()})


@app.route("/api/state", methods=["GET"])
def get_state():
    board = _board()
    return jsonify({
        "fen": board.fen(),
        "game_over": board.is_game_over(),
        "result": _get_result(board),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "is_check": board.is_check(),
        "move_count": len(board.move_stack),
    })


def _get_result(board):
    if not board.is_game_over():
        return None
    result = board.result()
    reason = "Checkmate" if board.is_checkmate() else "Resignation"
    if result == "1-0":
        return {"result": "1-0", "message": f"by {reason}"}
    elif result == "0-1":
        return {"result": "0-1", "message": f"by {reason}"}
    else:
        reason = "Draw"
        if board.is_stalemate():
            reason = "by Stalemate"
        elif board.is_insufficient_material():
            reason = "by Insufficient Material"
        elif board.can_claim_fifty_moves():
            reason = "by Fifty-Move Rule"
        elif board.can_claim_threefold_repetition():
            reason = "by Threefold Repetition"
        return {"result": "1/2-1/2", "message": reason}


if __name__ == "__main__":
    for d in (config.CHECKPOINT_DIR, config.DATA_RAW_DIR,
              config.DATA_PROC_DIR, config.RESULTS_DIR):
        os.makedirs(d, exist_ok=True)

    print("\n>> Chess AI Web Interface")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
