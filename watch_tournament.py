"""
Live tournament spectator — watch the Chess AI play Stockfish in your browser,
move by move, with a board, eval bar, move list and a running scoreboard.

Run:  python watch_tournament.py
Open: http://localhost:5001   (then click "Start tournament")
"""
import os, sys, json, time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from flask import Flask, Response, render_template_string
import chess

import config
# The viewer does light single-position inference; default it to CPU so it does not
# contend with a training run for the GPU (override with VIEWER_DEVICE=cuda).
config.DEVICE = os.environ.get("VIEWER_DEVICE", "cpu")
from src.engine import ChessAIEngine

try:
    from stockfish import Stockfish
    HAS_STOCKFISH = True
except ImportError:
    HAS_STOCKFISH = False

MOVE_DELAY   = 0.35          # seconds between moves so games are watchable
EVAL_DEPTH   = 8            # shallow Stockfish eval for the eval bar

app = Flask(__name__)
engine = ChessAIEngine()
config.TIME_LIMIT_SEC = config.TOURNAMENT_THINK_SEC   # snappy AI moves


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


def tournament_stream():
    if not HAS_STOCKFISH:
        yield _sse({"type": "error", "msg": "stockfish python package not installed"})
        return
    if not os.path.exists(config.STOCKFISH_PATH):
        yield _sse({"type": "error", "msg": f"Stockfish binary not found at {config.STOCKFISH_PATH}"})
        return

    levels = config.STOCKFISH_ELO_LEVELS
    games  = config.TOURNAMENT_GAMES
    standings = {elo: {"w": 0, "d": 0, "l": 0} for elo in levels}

    # Separate full-strength engine just for the eval bar
    sf_eval = Stockfish(path=config.STOCKFISH_PATH)
    sf_eval.set_depth(EVAL_DEPTH)

    yield _sse({"type": "init", "levels": levels, "games_per_level": games,
                "standings": standings})

    def eval_cp(board):
        try:
            sf_eval.set_fen_position(board.fen())
            ev = sf_eval.get_evaluation()           # White's perspective
            if ev["type"] == "cp":
                return max(-1500, min(1500, ev["value"]))
            # mate score → saturate the bar
            return 1500 if ev["value"] > 0 else -1500
        except Exception:
            return 0

    for elo in levels:
        sf = Stockfish(path=config.STOCKFISH_PATH,
                       parameters={"UCI_LimitStrength": True, "UCI_Elo": elo})
        for g in range(games):
            ai_white = (g % 2 == 0)
            board = chess.Board()

            yield _sse({"type": "matchup", "elo": elo, "game": g + 1,
                        "total": games, "ai_color": "white" if ai_white else "black",
                        "fen": board.fen()})

            while not board.is_game_over(claim_draw=True):
                ai_turn = (board.turn == chess.WHITE) == ai_white
                if ai_turn:
                    move, _ = engine.get_move_with_eval(board, use_book=True)
                    mover = "AI"
                else:
                    sf.set_fen_position(board.fen())
                    move = chess.Move.from_uci(sf.get_best_move_time(config.STOCKFISH_MOVETIME_MS))
                    mover = "SF"

                san = board.san(move)
                board.push(move)

                yield _sse({"type": "move", "fen": board.fen(), "san": san,
                            "uci": move.uci(), "mover": mover,
                            "from": chess.square_name(move.from_square),
                            "to": chess.square_name(move.to_square),
                            "ply": len(board.move_stack),
                            "eval": eval_cp(board)})
                time.sleep(MOVE_DELAY)

            result = board.result(claim_draw=True)
            ai_won = (result == "1-0" and ai_white) or (result == "0-1" and not ai_white)
            draw   = result == "1/2-1/2"
            if ai_won:   standings[elo]["w"] += 1
            elif draw:   standings[elo]["d"] += 1
            else:        standings[elo]["l"] += 1

            yield _sse({"type": "gameover", "elo": elo, "result": result,
                        "ai_won": ai_won, "draw": draw, "standings": standings})
            time.sleep(MOVE_DELAY * 2)

    yield _sse({"type": "done", "standings": standings})


@app.route("/stream")
def stream():
    return Response(tournament_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/")
def index():
    return render_template_string(PAGE)


PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chess AI — Live Tournament</title>
<style>
  :root{
    --bg:#0f1216; --panel:#171c22; --panel2:#1f262e; --line:#2a323c;
    --txt:#e7edf3; --muted:#8b97a5; --accent:#6ea8fe; --win:#4ade80;
    --loss:#f87171; --draw:#fbbf24; --light:#b9c2cc; --dark:#5b6b7a;
    --hl:rgba(110,168,254,.45);
  }
  *{box-sizing:border-box}
  body{margin:0;background:radial-gradient(1200px 600px at 50% -10%,#1a222b,#0f1216);
       color:var(--txt);font-family:Inter,Segoe UI,system-ui,sans-serif}
  header{padding:18px 26px;border-bottom:1px solid var(--line);display:flex;
         align-items:center;gap:16px}
  header h1{font-size:18px;margin:0;font-weight:650;letter-spacing:.3px}
  header .sub{color:var(--muted);font-size:13px}
  .wrap{display:grid;grid-template-columns:auto 360px;gap:26px;padding:26px;
        align-items:start;justify-content:center}
  .boardwrap{display:flex;gap:14px;align-items:stretch}
  .evalbar{width:18px;border-radius:9px;overflow:hidden;background:var(--dark);
           position:relative;border:1px solid var(--line)}
  .evalfill{position:absolute;left:0;right:0;bottom:0;background:var(--light);
            transition:height .3s ease}
  .board{width:560px;height:560px;display:grid;grid-template-columns:repeat(8,1fr);
         grid-template-rows:repeat(8,1fr);border-radius:10px;overflow:hidden;
         box-shadow:0 20px 60px rgba(0,0,0,.5)}
  .sq{display:flex;align-items:center;justify-content:center;font-size:42px;
      line-height:1;user-select:none;position:relative}
  .sq.l{background:#c9d3dd}.sq.d{background:#5c7488}
  .sq.hl::after{content:"";position:absolute;inset:0;background:var(--hl)}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;
         padding:18px}
  .matchup{font-size:15px;margin-bottom:6px}
  .tag{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;
       font-weight:600;margin-right:6px}
  .tag.ai{background:#1d3a6b;color:#cfe0ff}.tag.sf{background:#3a2330;color:#ffd6e6}
  .evalnum{font-variant-numeric:tabular-nums;color:var(--muted);font-size:13px;margin-top:4px}
  table{width:100%;border-collapse:collapse;margin-top:8px;font-size:13px}
  th,td{padding:7px 8px;text-align:center;border-bottom:1px solid var(--line)}
  th{color:var(--muted);font-weight:600}
  td.w{color:var(--win)}td.l{color:var(--loss)}td.d{color:var(--draw)}
  .moves{max-height:230px;overflow:auto;background:var(--panel2);border-radius:10px;
         padding:10px;font-variant-numeric:tabular-nums;font-size:13px;line-height:1.9;
         margin-top:6px}
  .moves .mv{display:inline-block;min-width:62px;color:var(--light)}
  .moves .no{color:var(--muted);display:inline-block;min-width:30px}
  button{background:var(--accent);color:#06101f;border:0;border-radius:10px;
         padding:10px 18px;font-weight:650;cursor:pointer;font-size:14px}
  button:disabled{opacity:.5;cursor:default}
  .status{color:var(--muted);font-size:13px;margin-left:auto}
  h3{margin:18px 0 4px;font-size:13px;color:var(--muted);text-transform:uppercase;
     letter-spacing:.6px;font-weight:600}
</style>
</head>
<body>
<header>
  <h1>♞ Chess AI — Live Tournament</h1>
  <span class="sub">CNN policy+value · iterative-deepening α-β · vs Stockfish</span>
  <span class="status" id="status">idle</span>
</header>

<div class="wrap">
  <div class="boardwrap">
    <div class="evalbar"><div class="evalfill" id="evalfill" style="height:50%"></div></div>
    <div class="board" id="board"></div>
  </div>

  <div>
    <div class="panel">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
        <button id="startBtn">Start tournament</button>
        <span class="evalnum" id="evalnum">eval 0.00</span>
      </div>
      <div class="matchup" id="matchup">Press start to begin.</div>
      <h3>Scoreboard</h3>
      <table id="score">
        <thead><tr><th>Stockfish</th><th>W</th><th>D</th><th>L</th><th>Score</th></tr></thead>
        <tbody></tbody>
      </table>
      <h3>Moves</h3>
      <div class="moves" id="moves"></div>
    </div>
  </div>
</div>

<script>
const PIECES = {p:'♟',n:'♞',b:'♝',r:'♜',q:'♛',k:'♚',
                P:'♙',N:'♘',B:'♗',R:'♖',Q:'♕',K:'♔'};
const boardEl = document.getElementById('board');
const movesEl = document.getElementById('moves');
const statusEl= document.getElementById('status');
let sqEls = [];

function buildBoard(){
  boardEl.innerHTML='';
  sqEls=[];
  for(let r=0;r<8;r++){
    for(let f=0;f<8;f++){
      const d=document.createElement('div');
      d.className='sq '+(((r+f)%2===0)?'l':'d');
      boardEl.appendChild(d);
      sqEls.push(d);
    }
  }
}
function nameToIdx(name){
  const f=name.charCodeAt(0)-97, r=8-parseInt(name[1]);
  return r*8+f;
}
function render(fen, from, to){
  const rows=fen.split(' ')[0].split('/');
  let i=0;
  for(let r=0;r<8;r++){
    let f=0;
    for(const c of rows[r]){
      if(/\d/.test(c)){ for(let k=0;k<+c;k++){sqEls[r*8+f].textContent='';f++;} }
      else { sqEls[r*8+f].textContent=PIECES[c]||''; f++; }
    }
  }
  sqEls.forEach(s=>s.classList.remove('hl'));
  if(from) sqEls[nameToIdx(from)].classList.add('hl');
  if(to)   sqEls[nameToIdx(to)].classList.add('hl');
}
function setEval(cp){
  const pct=Math.max(2,Math.min(98, 50 + (cp/1500)*50));
  document.getElementById('evalfill').style.height=pct+'%';
  document.getElementById('evalnum').textContent='eval '+(cp/100).toFixed(2);
}
function scoreRow(elo,s){
  const n=s.w+s.d+s.l;
  const pct=n? (((s.w+s.d*0.5)/n)*100).toFixed(0):'0';
  return `<tr><td>${elo}</td><td class="w">${s.w}</td><td class="d">${s.d}</td>`+
         `<td class="l">${s.l}</td><td>${pct}%</td></tr>`;
}
function renderScore(st){
  const tb=document.querySelector('#score tbody');
  tb.innerHTML=Object.keys(st).map(e=>scoreRow(e,st[e])).join('');
}

buildBoard();
render(new Array(8).fill('8').join('/')+' w','','');
render('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w','','');

let plyNo=0;
document.getElementById('startBtn').addEventListener('click', ()=>{
  document.getElementById('startBtn').disabled=true;
  movesEl.innerHTML=''; plyNo=0;
  const es=new EventSource('/stream');
  es.onmessage=(e)=>{
    const m=JSON.parse(e.data);
    if(m.type==='init'){ renderScore(m.standings); statusEl.textContent='running…'; }
    else if(m.type==='matchup'){
      plyNo=0; movesEl.innerHTML='';
      render(m.fen,'','');
      const aiW=m.ai_color==='white';
      document.getElementById('matchup').innerHTML=
        `<span class="tag ai">AI (${m.ai_color})</span> vs `+
        `<span class="tag sf">Stockfish ${m.elo}</span> · game ${m.game}/${m.total}`;
    }
    else if(m.type==='move'){
      render(m.fen,m.from,m.to);
      setEval(m.eval);
      if(m.ply%2===1){ plyNo++; movesEl.innerHTML+=`<span class="no">${plyNo}.</span>`; }
      movesEl.innerHTML+=`<span class="mv" title="${m.mover}">${m.san}</span>`;
      movesEl.scrollTop=movesEl.scrollHeight;
    }
    else if(m.type==='gameover'){
      renderScore(m.standings);
      const r=m.ai_won?'AI wins':(m.draw?'draw':'Stockfish wins');
      movesEl.innerHTML+=`<span class="no"></span><b>· ${m.result} (${r})</b><br>`;
    }
    else if(m.type==='done'){ statusEl.textContent='tournament complete'; es.close();
      document.getElementById('startBtn').disabled=false;
      document.getElementById('startBtn').textContent='Run again'; }
    else if(m.type==='error'){ statusEl.textContent='error: '+m.msg; es.close(); }
  };
  es.onerror=()=>{ statusEl.textContent='stream ended'; es.close();
    document.getElementById('startBtn').disabled=false; };
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("\n>> Live Tournament Viewer")
    print("   Open http://localhost:5001 in your browser, then click Start.\n")
    # threaded=True so the SSE stream and static requests don't block each other
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
