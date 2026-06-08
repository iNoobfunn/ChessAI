"""
Flask web server for Chess AI — play against the bot in a browser.
Run: python app.py
Then open http://localhost:5000
"""
import os, sys, json

# Ensure chess_ai is the working context
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import chess

import config
from src.model import load_model
from src.inference import predict_move
from src.minimax import search

app = Flask(__name__, template_folder="web", static_folder="web/static")
CORS(app)

# ── Load the AI engine ──────────────────────────────────
checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best.pt")
DEVICE = config.DEVICE
model = load_model(checkpoint_path, DEVICE)
print(f"[OK] Chess AI model loaded on {DEVICE}")

# ── Game state (per-session, single player for simplicity) ──
game_board = chess.Board()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/new_game", methods=["POST"])
def new_game():
    global game_board
    data = request.get_json() or {}
    player_color = data.get("player_color", "white")
    game_board = chess.Board()

    response = {
        "fen": game_board.fen(),
        "player_color": player_color,
        "game_over": False,
        "result": None,
        "ai_move": None,
    }

    # If player chose black, AI makes the first move
    if player_color == "black":
        ai_move, raw_score = _get_ai_move()
        game_board.push(ai_move)
        response["fen"] = game_board.fen()
        response["ai_move"] = ai_move.uci()
        response["eval_score"] = round(raw_score / 100.0, 2)  # AI is White, so positive means White winning

    return jsonify(response)


@app.route("/api/move", methods=["POST"])
def make_move():
    global game_board
    data = request.get_json()
    uci_move = data.get("move")

    try:
        move = chess.Move.from_uci(uci_move)
    except Exception:
        return jsonify({"error": "Invalid move format"}), 400

    # Handle promotion: if a pawn reaches the last rank, default to queen
    if game_board.piece_at(move.from_square) and \
       game_board.piece_at(move.from_square).piece_type == chess.PAWN:
        if chess.square_rank(move.to_square) in (0, 7) and move.promotion is None:
            move = chess.Move(move.from_square, move.to_square, chess.QUEEN)

    if move not in game_board.legal_moves:
        return jsonify({"error": "Illegal move", "legal_moves": [m.uci() for m in game_board.legal_moves]}), 400

    # Make player's move
    san = game_board.san(move)
    game_board.push(move)

    response = {
        "fen": game_board.fen(),
        "player_move_san": san,
        "game_over": game_board.is_game_over(),
        "result": _get_result(),
        "ai_move": None,
        "ai_move_san": None,
    }

    # If game isn't over, let AI respond
    if not game_board.is_game_over():
        try:
            ai_move, raw_score = _get_ai_move()
            ai_san = game_board.san(ai_move)
            
            ai_is_white = game_board.turn == chess.WHITE
            std_score = raw_score if ai_is_white else -raw_score
            
            game_board.push(ai_move)
            response["fen"] = game_board.fen()
            response["ai_move"] = ai_move.uci()
            response["ai_move_san"] = ai_san
            response["eval_score"] = round(std_score / 100.0, 2)
            response["game_over"] = game_board.is_game_over()
            response["result"] = _get_result()
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"AI crashed: {str(e)}"}), 500

    return jsonify(response)


@app.route("/api/legal_moves", methods=["GET"])
def legal_moves():
    moves = []
    for m in game_board.legal_moves:
        moves.append({
            "uci": m.uci(),
            "from": chess.square_name(m.from_square),
            "to": chess.square_name(m.to_square),
        })
    return jsonify({"moves": moves, "fen": game_board.fen()})


@app.route("/api/undo", methods=["POST"])
def undo():
    global game_board
    # Undo both AI and player moves
    if len(game_board.move_stack) >= 2:
        game_board.pop()  # undo AI
        game_board.pop()  # undo player
    elif len(game_board.move_stack) == 1:
        game_board.pop()
    return jsonify({"fen": game_board.fen(), "game_over": game_board.is_game_over()})


@app.route("/api/state", methods=["GET"])
def get_state():
    return jsonify({
        "fen": game_board.fen(),
        "game_over": game_board.is_game_over(),
        "result": _get_result(),
        "turn": "white" if game_board.turn == chess.WHITE else "black",
        "is_check": game_board.is_check(),
        "move_count": len(game_board.move_stack),
    })


def _get_ai_move():
    """Get AI's move — tries minimax first, falls back to pure prediction."""
    try:
        return search(game_board, model, DEVICE, depth=config.SEARCH_DEPTH,
                      time_limit=config.TIME_LIMIT_SEC)
    except Exception as e:
        print(f"Search failed: {e}, falling back to direct prediction")
        return predict_move(game_board, model, DEVICE)


def _get_result():
    if not game_board.is_game_over():
        return None
    result = game_board.result()
    if result == "1-0":
        return {"result": "1-0", "message": "White wins!"}
    elif result == "0-1":
        return {"result": "0-1", "message": "Black wins!"}
    else:
        reason = "Draw"
        if game_board.is_stalemate():
            reason = "Stalemate"
        elif game_board.is_insufficient_material():
            reason = "Insufficient material"
        elif game_board.can_claim_fifty_moves():
            reason = "Fifty-move rule"
        elif game_board.can_claim_threefold_repetition():
            reason = "Threefold repetition"
        return {"result": "1/2-1/2", "message": reason}


if __name__ == "__main__":
    # Create required directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.DATA_RAW_DIR, exist_ok=True)
    os.makedirs(config.DATA_PROC_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("\n>> Chess AI Web Interface")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
