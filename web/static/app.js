/* ═══════════════════════════════════════════════════════
   Chess AI — Interactive Board Controller
   ═══════════════════════════════════════════════════════ */

// ── Piece SVG Map ────────────────────────────────────────
const PIECE_SVG_URL = {
    'K': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wk.png',
    'Q': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wq.png',
    'R': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wr.png',
    'B': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wb.png',
    'N': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wn.png',
    'P': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wp.png',
    'k': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bk.png',
    'q': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bq.png',
    'r': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/br.png',
    'b': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bb.png',
    'n': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bn.png',
    'p': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bp.png',
};

// ── State ────────────────────────────────────────────────
let boardState = [];          // 8x8 array of piece chars or null
let selectedSquare = null;     // {row, col} or null
let legalMoves = [];           // [{from, to, uci}]
let playerColor = 'white';     // 'white' or 'black'
let boardFlipped = false;      // is board visually flipped?
let isThinking = false;        // is AI currently thinking?
let lastMove = null;           // {from: {row, col}, to: {row, col}}
let moveList = [];             // [{number, white, black}]
let dragPiece = null;          // dragging state
let gameActive = false;
let checkSquare = null;        // {row, col} of king in check

// ── Clock State ──────────────────────────────────────────
let timeControlMs = 10 * 60 * 1000;
let topTimeMs = timeControlMs;
let bottomTimeMs = timeControlMs;
let clockInterval = null;
let lastTickTime = null;
// ── Initialization ───────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initBoard();
    fetchState();
    // Show new game dialog on first load
    setTimeout(() => showNewGameDialog(), 300);
});

// ── Board Rendering ──────────────────────────────────────
function initBoard() {
    const board = document.getElementById('chessBoard');
    board.innerHTML = '';

    for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 8; col++) {
            const sq = document.createElement('div');
            sq.className = `square ${(row + col) % 2 === 0 ? 'light' : 'dark'}`;
            sq.dataset.row = row;
            sq.dataset.col = col;
            sq.id = `sq-${row}-${col}`;
            sq.addEventListener('click', () => onSquareClick(row, col));
            sq.addEventListener('mousedown', (e) => onPieceMouseDown(e, row, col));
            board.appendChild(sq);
        }
    }

    // Drag handlers on document level
    document.addEventListener('mousemove', onPieceMouseMove);
    document.addEventListener('mouseup', onPieceMouseUp);

    updateLabels();
}

function updateLabels() {
    const rankLabels = document.getElementById('rankLabels');
    const fileLabels = document.getElementById('fileLabels');
    rankLabels.innerHTML = '';
    fileLabels.innerHTML = '';

    for (let i = 0; i < 8; i++) {
        const rank = document.createElement('div');
        rank.textContent = boardFlipped ? (i + 1) : (8 - i);
        rankLabels.appendChild(rank);

        const file = document.createElement('div');
        file.textContent = boardFlipped ? 'hgfedcba'[i] : 'abcdefgh'[i];
        fileLabels.appendChild(file);
    }
}

function renderBoard() {
    for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 8; col++) {
            const sq = document.getElementById(`sq-${row}-${col}`);
            
            // Reset classes
            sq.className = `square ${(row + col) % 2 === 0 ? 'light' : 'dark'}`;

            // Remove old piece
            const oldPiece = sq.querySelector('.piece');
            if (oldPiece) oldPiece.remove();

            // Get actual board position (accounting for flip)
            const actualRow = boardFlipped ? (7 - row) : row;
            const actualCol = boardFlipped ? (7 - col) : col;
            const piece = boardState[actualRow]?.[actualCol];

            if (piece) {
                const pieceEl = document.createElement('div');
                pieceEl.className = `piece ${piece === piece.toUpperCase() ? 'white-piece' : 'black-piece'}`;
                pieceEl.style.backgroundImage = `url("${PIECE_SVG_URL[piece]}")`;
                pieceEl.dataset.piece = piece;
                sq.appendChild(pieceEl);
            }

            // Highlight last move
            if (lastMove) {
                const dispFrom = toDisplayCoords(lastMove.from.row, lastMove.from.col);
                const dispTo = toDisplayCoords(lastMove.to.row, lastMove.to.col);
                if (row === dispFrom.row && col === dispFrom.col) sq.classList.add('last-move');
                if (row === dispTo.row && col === dispTo.col) sq.classList.add('last-move');
            }

            // Highlight selected
            if (selectedSquare) {
                const dispSel = toDisplayCoords(selectedSquare.row, selectedSquare.col);
                if (row === dispSel.row && col === dispSel.col) {
                    sq.classList.add('selected');
                }
            }

            // Show legal move dots
            if (selectedSquare) {
                const movesForSelected = getLegalMovesFrom(selectedSquare.row, selectedSquare.col);
                for (const m of movesForSelected) {
                    const dispM = toDisplayCoords(m.toRow, m.toCol);
                    if (row === dispM.row && col === dispM.col) {
                        const targetPiece = boardState[m.toRow]?.[m.toCol];
                        sq.classList.add(targetPiece ? 'legal-capture' : 'legal-move');
                    }
                }
            }

            // Check highlight
            if (checkSquare) {
                const dispCheck = toDisplayCoords(checkSquare.row, checkSquare.col);
                if (row === dispCheck.row && col === dispCheck.col) {
                    sq.classList.add('in-check');
                }
            }
        }
    }

    calculateMaterial();
}

function calculateMaterial() {
    const pieceValues = { 'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9 };
    const startingCounts = { 'p': 8, 'n': 2, 'b': 2, 'r': 2, 'q': 1, 'P': 8, 'N': 2, 'B': 2, 'R': 2, 'Q': 1 };
    
    let currentCounts = { 'p': 0, 'n': 0, 'b': 0, 'r': 0, 'q': 0, 'P': 0, 'N': 0, 'B': 0, 'R': 0, 'Q': 0 };
    let whiteScore = 0;
    let blackScore = 0;

    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            const p = boardState[r]?.[c];
            if (p && currentCounts[p] !== undefined) {
                currentCounts[p]++;
                if (p === p.toUpperCase()) {
                    whiteScore += pieceValues[p];
                } else {
                    blackScore += pieceValues[p];
                }
            }
        }
    }

    const whiteCaptured = [];
    const blackCaptured = [];
    
    // Pieces white has captured (black pieces missing)
    ['q', 'r', 'b', 'n', 'p'].forEach(p => {
        const missing = startingCounts[p] - currentCounts[p];
        for (let i = 0; i < missing; i++) whiteCaptured.push(p);
    });

    // Pieces black has captured (white pieces missing)
    ['Q', 'R', 'B', 'N', 'P'].forEach(p => {
        const missing = startingCounts[p] - currentCounts[p];
        for (let i = 0; i < missing; i++) blackCaptured.push(p);
    });

    const pieceIcons = {
        'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕',
        'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛'
    };

    const renderMaterial = (elementId, captured, isWhite) => {
        const el = document.getElementById(elementId);
        if (!el) return;
        el.innerHTML = '';
        
        for (const p of captured) {
            const span = document.createElement('span');
            span.className = 'captured-piece-icon';
            span.textContent = pieceIcons[p];
            el.appendChild(span);
        }

        const scoreDiff = isWhite ? (whiteScore - blackScore) : (blackScore - whiteScore);
        if (scoreDiff > 0) {
            const diffSpan = document.createElement('span');
            diffSpan.className = 'material-score';
            diffSpan.textContent = `+${scoreDiff}`;
            el.appendChild(diffSpan);
        }
    };

    const playerIsWhite = playerColor === 'white';
    renderMaterial('bottomMaterial', playerIsWhite ? whiteCaptured : blackCaptured, playerIsWhite);
    renderMaterial('topMaterial', playerIsWhite ? blackCaptured : whiteCaptured, !playerIsWhite);
}


function toDisplayCoords(row, col) {
    if (boardFlipped) return { row: 7 - row, col: 7 - col };
    return { row, col };
}

function fromDisplayCoords(row, col) {
    if (boardFlipped) return { row: 7 - row, col: 7 - col };
    return { row, col };
}

// ── FEN Parsing ──────────────────────────────────────────
function parseFEN(fen) {
    const parts = fen.split(' ');
    const rows = parts[0].split('/');
    const board = [];

    for (let r = 0; r < 8; r++) {
        board[r] = [];
        let col = 0;
        for (const ch of rows[r]) {
            if (ch >= '1' && ch <= '8') {
                for (let i = 0; i < parseInt(ch); i++) {
                    board[r][col++] = null;
                }
            } else {
                board[r][col++] = ch;
            }
        }
    }
    return board;
}

function findKing(color) {
    const kingChar = color === 'white' ? 'K' : 'k';
    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            if (boardState[r]?.[c] === kingChar) return { row: r, col: c };
        }
    }
    return null;
}

// ── Square/Move Helpers ──────────────────────────────────
function squareToAlgebraic(row, col) {
    return 'abcdefgh'[col] + (8 - row);
}

function algebraicToSquare(sq) {
    return { row: 8 - parseInt(sq[1]), col: 'abcdefgh'.indexOf(sq[0]) };
}

function getLegalMovesFrom(row, col) {
    const sqName = squareToAlgebraic(row, col);
    return legalMoves
        .filter(m => m.from === sqName)
        .map(m => {
            const to = algebraicToSquare(m.to);
            return { toRow: to.row, toCol: to.col, uci: m.uci };
        });
}

function isPlayerPiece(row, col) {
    const piece = boardState[row]?.[col];
    if (!piece) return false;
    if (playerColor === 'white') return piece === piece.toUpperCase();
    return piece === piece.toLowerCase();
}

// ── Click & Drag Interaction ─────────────────────────────
function onSquareClick(displayRow, displayCol) {
    if (isThinking || !gameActive) return;

    const actual = fromDisplayCoords(displayRow, displayCol);
    const row = actual.row;
    const col = actual.col;

    if (selectedSquare) {
        // Try to make a move
        const movesFrom = getLegalMovesFrom(selectedSquare.row, selectedSquare.col);
        const targetMoves = movesFrom.filter(m => m.toRow === row && m.toCol === col);

        if (targetMoves.length > 0) {
            if (targetMoves.length > 1 || targetMoves[0].uci.length > 4) {
                showPromotionDialog(targetMoves, {row, col}, selectedSquare);
                return;
            }
            const targetMove = targetMoves[0];
            // Optimistic move
            boardState[row][col] = boardState[selectedSquare.row][selectedSquare.col];
            boardState[selectedSquare.row][selectedSquare.col] = null;
            
            selectedSquare = null;
            renderBoard(); // Render optimistic state immediately for zero delay
            
            makeMove(targetMove.uci);
        } else if (isPlayerPiece(row, col)) {
            // Select different piece
            selectedSquare = { row, col };
        } else {
            selectedSquare = null;
        }
    } else {
        // Select piece
        if (isPlayerPiece(row, col)) {
            selectedSquare = { row, col };
        }
    }

    renderBoard();
}

function onPieceMouseDown(e, displayRow, displayCol) {
    if (isThinking || !gameActive) return;

    const actual = fromDisplayCoords(displayRow, displayCol);
    if (!isPlayerPiece(actual.row, actual.col)) return;

    e.preventDefault();
    selectedSquare = { row: actual.row, col: actual.col };

    const sq = document.getElementById(`sq-${displayRow}-${displayCol}`);
    const pieceEl = sq.querySelector('.piece');
    if (!pieceEl) return;

    dragPiece = {
        element: pieceEl,
        fromRow: actual.row,
        fromCol: actual.col,
        startX: e.clientX,
        startY: e.clientY,
    };

    pieceEl.classList.add('dragging');
    pieceEl.style.position = 'fixed';
    
    // Get square size
    const sqSize = sq.getBoundingClientRect().width;
    
    pieceEl.style.left = (e.clientX - sqSize/2) + 'px';
    pieceEl.style.top = (e.clientY - sqSize/2) + 'px';
    pieceEl.style.width = sqSize + 'px';
    pieceEl.style.height = sqSize + 'px';
    pieceEl.style.pointerEvents = 'none';
    document.body.appendChild(pieceEl);

    renderBoard();
}

function onPieceMouseMove(e) {
    if (!dragPiece) return;
    const sqSize = parseFloat(dragPiece.element.style.width);
    dragPiece.element.style.left = (e.clientX - sqSize/2) + 'px';
    dragPiece.element.style.top = (e.clientY - sqSize/2) + 'px';
}

function onPieceMouseUp(e) {
    if (!dragPiece) return;

    const el = dragPiece.element;
    el.classList.remove('dragging');
    el.style.position = '';
    el.style.left = '';
    el.style.top = '';
    el.style.width = '';
    el.style.height = '';
    el.style.pointerEvents = '';

    // Find which square we dropped on
    el.remove(); // temporarily remove to find element underneath
    const dropTarget = document.elementFromPoint(e.clientX, e.clientY);

    const sq = dropTarget?.closest('.square');
    if (sq) {
        const dispRow = parseInt(sq.dataset.row);
        const dispCol = parseInt(sq.dataset.col);
        const actual = fromDisplayCoords(dispRow, dispCol);

        const movesFrom = getLegalMovesFrom(dragPiece.fromRow, dragPiece.fromCol);
        const targetMoves = movesFrom.filter(m => m.toRow === actual.row && m.toCol === actual.col);

        if (targetMoves.length > 0) {
            if (targetMoves.length > 1 || targetMoves[0].uci.length > 4) {
                // Keep piece visually on target square during promotion dialog
                dragPiece.element.classList.remove('dragging');
                dragPiece.element.style.left = '0';
                dragPiece.element.style.top = '0';
                dragPiece.element.style.position = '';
                sq.appendChild(dragPiece.element);
                
                showPromotionDialog(targetMoves, actual, {row: dragPiece.fromRow, col: dragPiece.fromCol});
                return;
            }
            const targetMove = targetMoves[0];
            // Optimistic move
            boardState[actual.row][actual.col] = boardState[dragPiece.fromRow][dragPiece.fromCol];
            boardState[dragPiece.fromRow][dragPiece.fromCol] = null;
            
            dragPiece = null;
            selectedSquare = null;
            renderBoard(); // Render optimistic state immediately for zero delay
            
            makeMove(targetMove.uci);
            return;
        }
    }

    dragPiece = null;
    renderBoard();
}

// ── API Calls ────────────────────────────────────────────
async function fetchState() {
    try {
        const res = await fetch('/api/state');
        const data = await res.json();
        boardState = parseFEN(data.fen);
        checkSquare = data.is_check ? findKing(data.turn) : null;
        renderBoard();
    } catch (e) {
        console.error('Failed to fetch state:', e);
    }
}

async function fetchLegalMoves() {
    try {
        const res = await fetch('/api/legal_moves');
        const data = await res.json();
        legalMoves = data.moves;
        boardState = parseFEN(data.fen);
    } catch (e) {
        console.error('Failed to fetch legal moves:', e);
    }
}

async function makeMove(uci) {
    if (isThinking) return;

    setThinking(true);

    try {
        const res = await fetch('/api/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ move: uci }),
        });

        if (!res.ok) {
            let errStr = "Unknown error";
            try {
                const err = await res.json();
                errStr = err.error || errStr;
            } catch (e) {
                errStr = await res.text();
            }
            console.error('Move error:', errStr);
            alert("AI/Backend Error: " + errStr);
            setThinking(false);
            await fetchLegalMoves();
            renderBoard();
            return;
        }

        const data = await res.json();
        
        // Update board after player move
        boardState = parseFEN(data.fen);

        // Record player move
        const playerSan = data.player_move_san;
        addMoveToHistory(playerSan, playerColor);

        // Record last move (AI's response)
        if (data.ai_move) {
            const aiFrom = algebraicToSquare(data.ai_move.substring(0, 2));
            const aiTo = algebraicToSquare(data.ai_move.substring(2, 4));
            lastMove = { from: aiFrom, to: aiTo };
            addMoveToHistory(data.ai_move_san, playerColor === 'white' ? 'black' : 'white');
        } else {
            // Record player's last move
            const pFrom = algebraicToSquare(uci.substring(0, 2));
            const pTo = algebraicToSquare(uci.substring(2, 4));
            lastMove = { from: pFrom, to: pTo };
        }

        // Check for check state
        const stateRes = await fetch('/api/state');
        const stateData = await stateRes.json();
        checkSquare = stateData.is_check ? findKing(stateData.turn) : null;

        // Refresh legal moves
        await fetchLegalMoves();
        renderBoard();

        // Check game over
        if (data.game_over && data.result) {
            gameActive = false;
            if (clockInterval) clearInterval(clockInterval);
            updateClockDisplays();
            showGameOverDialog(data.result);
        }

        if (data.eval_score !== undefined) {
            updateEvalBar(data.eval_score);
        }

    } catch (e) {
        console.error('Move failed:', e);
    }

    setThinking(false);
}

async function startNewGame() {
    const selectedColor = document.querySelector('.color-option.selected');
    playerColor = selectedColor?.id === 'optBlack' ? 'black' : 'white';

    hideNewGameDialog();
    if (clockInterval) clearInterval(clockInterval);
    topTimeMs = timeControlMs;
    bottomTimeMs = timeControlMs;
    updateClockDisplays();
    setThinking(true);

    try {
        const res = await fetch('/api/new_game', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ player_color: playerColor }),
        });

        const data = await res.json();
        boardState = parseFEN(data.fen);
        selectedSquare = null;
        lastMove = null;
        checkSquare = null;
        moveList = [];
        gameActive = true;
        
        lastTickTime = Date.now();
        clockInterval = setInterval(timerLoop, 100);

        // If AI made first move (player is black)
        if (data.ai_move) {
            const aiFrom = algebraicToSquare(data.ai_move.substring(0, 2));
            const aiTo = algebraicToSquare(data.ai_move.substring(2, 4));
            lastMove = { from: aiFrom, to: aiTo };
            // We need to get the SAN for the AI move
            addMoveToHistory(data.ai_move, 'white');
        }

        // Auto-flip board if playing black
        boardFlipped = (playerColor === 'black');
        updateLabels();

        // Update player label
        document.getElementById('bottomDetail').textContent = 
            `Playing as ${playerColor.charAt(0).toUpperCase() + playerColor.slice(1)}`;

        // Update active player card
        updatePlayerCards();

        await fetchLegalMoves();
        renderMoveHistory();
        renderBoard();

        if (data.eval_score !== undefined) {
            updateEvalBar(data.eval_score);
        } else {
            updateEvalBar(0.00);
        }

    } catch (e) {
        console.error('New game failed:', e);
    }

    setThinking(false);
}

async function undoMove() {
    if (isThinking) return;

    try {
        const res = await fetch('/api/undo', { method: 'POST' });
        const data = await res.json();
        boardState = parseFEN(data.fen);
        selectedSquare = null;
        lastMove = null;
        checkSquare = null;
        gameActive = !data.game_over;

        // Remove last two entries from move history
        if (moveList.length > 0) {
            const last = moveList[moveList.length - 1];
            if (last.black) {
                last.black = null;
            } else {
                moveList.pop();
            }
            if (moveList.length > 0) {
                const prev = moveList[moveList.length - 1];
                if (prev.black) {
                    prev.black = null;
                } else {
                    moveList.pop();
                }
            }
        }

        await fetchLegalMoves();
        renderMoveHistory();
        renderBoard();
    } catch (e) {
        console.error('Undo failed:', e);
    }
}

function flipBoard() {
    boardFlipped = !boardFlipped;
    updateLabels();
    renderBoard();
    
    // Update eval bar rotation based on flip
    const evalBar = document.getElementById('evalBar');
    const evalText = document.getElementById('evalText');
    if (evalBar && evalText) {
        if (boardFlipped) {
            evalBar.style.transform = 'rotate(180deg)';
            evalText.style.transform = 'rotate(180deg)';
        } else {
            evalBar.style.transform = 'none';
            evalText.style.transform = 'none';
        }
    }
}

// ── Move History Management ──────────────────────────────
function addMoveToHistory(san, color) {
    if (color === 'white') {
        moveList.push({ number: moveList.length + 1, white: san, black: null });
    } else {
        if (moveList.length === 0) {
            moveList.push({ number: 1, white: '...', black: san });
        } else {
            moveList[moveList.length - 1].black = san;
        }
    }
    renderMoveHistory();
}

function renderMoveHistory() {
    const container = document.getElementById('moveHistory');
    const counter = document.getElementById('moveCounter');

    if (moveList.length === 0) {
        container.innerHTML = '<div class="move-placeholder">Start a new game to begin...</div>';
        counter.textContent = '0 moves';
        return;
    }

    let totalMoves = 0;
    let html = '';
    for (const m of moveList) {
        const isLast = m === moveList[moveList.length - 1];
        html += `<div class="move-row fade-in">`;
        html += `<span class="move-number">${m.number}.</span>`;
        html += `<span class="move-white ${isLast && !m.black ? 'latest' : ''}">${m.white || ''}</span>`;
        if (m.black) {
            html += `<span class="move-black ${isLast ? 'latest' : ''}">${m.black}</span>`;
            totalMoves += 2;
        } else {
            html += `<span class="move-black"></span>`;
            totalMoves += 1;
        }
        html += `</div>`;
    }

    container.innerHTML = html;
    counter.textContent = `${totalMoves} moves`;
    container.scrollTop = container.scrollHeight;
}

// ── UI Updates ───────────────────────────────────────────
function setThinking(thinking) {
    isThinking = thinking;
    const overlay = document.getElementById('thinkingOverlay');
    const badge = document.getElementById('statusBadge');
    
    if (thinking) {
        overlay.classList.add('visible');
        badge.classList.add('thinking');
        badge.querySelector('.status-text').textContent = 'Thinking...';
    } else {
        overlay.classList.remove('visible');
        badge.classList.remove('thinking');
        badge.querySelector('.status-text').textContent = 'Ready';
    }

    updatePlayerCards();
}

function updatePlayerCards() {
    const aiCard = document.getElementById('topPlayerCard');
    const humanCard = document.getElementById('bottomPlayerCard');
    
    // Simple active state based on thinking
    if (isThinking) {
        aiCard.classList.add('active');
        humanCard.classList.remove('active');
    } else if (gameActive) {
        aiCard.classList.remove('active');
        humanCard.classList.add('active');
    } else {
        aiCard.classList.remove('active');
        humanCard.classList.remove('active');
    }
}

function updateEvalBar(score) {
    const evalFill = document.getElementById('evalFill');
    const evalText = document.getElementById('evalText');
    if (!evalFill || !evalText) return;

    let percentage = 50 + (score / 10) * 50;
    percentage = Math.max(5, Math.min(95, percentage));
    
    evalFill.style.height = `${percentage}%`;

    let text = score > 0 ? `+${score.toFixed(2)}` : score.toFixed(2);
    if (score === 0) text = "0.00";
    evalText.textContent = text;
    
    if (percentage >= 50) {
        evalText.style.bottom = '10px';
        evalText.style.top = 'auto';
        evalText.style.color = '#333';
    } else {
        evalText.style.top = '10px';
        evalText.style.bottom = 'auto';
        evalText.style.color = '#e2e2e2';
    }
}

// ── Dialogs ──────────────────────────────────────────────
function showNewGameDialog() {
    document.getElementById('newGameModal').classList.add('visible');
}

function hideNewGameDialog() {
    document.getElementById('newGameModal').classList.remove('visible');
}

function selectColor(color) {
    document.querySelectorAll('.color-option').forEach(el => el.classList.remove('selected'));
    document.getElementById(color === 'white' ? 'optWhite' : 'optBlack').classList.add('selected');
}

function showGameOverDialog(result) {
    const modal = document.getElementById('gameOverModal');
    const icon = document.getElementById('gameOverIcon');
    const title = document.getElementById('gameOverTitle');
    const message = document.getElementById('gameOverMessage');

    let playerWon = false;
    if (result.result === '1-0') playerWon = (playerColor === 'white');
    if (result.result === '0-1') playerWon = (playerColor === 'black');
    const isDraw = result.result === '1/2-1/2';

    if (isDraw) {
        icon.textContent = '🤝';
        title.textContent = 'Draw!';
        message.textContent = result.message;
    } else if (playerWon) {
        icon.textContent = '🏆';
        title.textContent = 'You Win!';
        message.textContent = 'Congratulations! You defeated the AI.';
    } else {
        icon.textContent = '♛';
        title.textContent = 'AI Wins';
        message.textContent = 'The neural network prevails. Try again!';
    }

    modal.classList.add('visible');
}

function hideGameOverDialog() {
    document.getElementById('gameOverModal').classList.remove('visible');
}

function toggleEvalBar() {
    const isChecked = document.getElementById('toggleEval').checked;
    const evalContainer = document.querySelector('.eval-bar-container');
    if (evalContainer) {
        evalContainer.style.display = isChecked ? 'block' : 'none';
    }
}

let pendingPromotion = null;

function showPromotionDialog(moves, toActual, fromActual) {
    pendingPromotion = { moves, toActual, fromActual };
    const modal = document.getElementById('promotionModal');
    const optionsContainer = document.getElementById('promotionOptions');
    optionsContainer.innerHTML = '';
    
    const isWhite = playerColor === 'white';
    const pieces = isWhite ? ['Q', 'R', 'B', 'N'] : ['q', 'r', 'b', 'n'];
    const uciChars = ['q', 'r', 'b', 'n'];
    
    pieces.forEach((p, idx) => {
        const div = document.createElement('div');
        div.className = 'promo-piece';
        div.style.backgroundImage = `url("${PIECE_SVG_URL[p]}")`;
        div.onclick = () => selectPromotion(uciChars[idx]);
        optionsContainer.appendChild(div);
    });
    
    modal.classList.add('visible');
}

function selectPromotion(promoChar) {
    document.getElementById('promotionModal').classList.remove('visible');
    
    if (!pendingPromotion) return;
    const { moves, toActual, fromActual } = pendingPromotion;
    
    const targetMove = moves.find(m => m.uci.endsWith(promoChar));
    
    if (targetMove) {
        const isWhite = playerColor === 'white';
        const promoPiece = isWhite ? promoChar.toUpperCase() : promoChar;
        boardState[toActual.row][toActual.col] = promoPiece;
        boardState[fromActual.row][fromActual.col] = null;
        
        selectedSquare = null;
        if (dragPiece) {
            dragPiece.element.remove();
            dragPiece = null;
        }
        renderBoard();
        
        makeMove(targetMove.uci);
    }
    
    pendingPromotion = null;
}

// ── Clock Logic ──────────────────────────────────────────

function selectTime(minutes) {
    document.querySelectorAll('.time-option').forEach(el => el.classList.remove('selected'));
    document.getElementById(minutes === 0 ? 'timeNone' : `time${minutes}`).classList.add('selected');
    timeControlMs = minutes * 60 * 1000;
}

function timerLoop() {
    if (!gameActive || timeControlMs === 0) return;

    const now = Date.now();
    const dt = now - lastTickTime;
    lastTickTime = now;

    if (isThinking) {
        topTimeMs = Math.max(0, topTimeMs - dt);
        if (topTimeMs === 0) handleTimeout('top');
    } else {
        bottomTimeMs = Math.max(0, bottomTimeMs - dt);
        if (bottomTimeMs === 0) handleTimeout('bottom');
    }
    
    updateClockDisplays();
}

function updateClockDisplays() {
    const topEl = document.getElementById('topCaptured');
    const bottomEl = document.getElementById('bottomCaptured');
    if (!topEl || !bottomEl) return;
    
    if (timeControlMs === 0) {
        topEl.style.display = 'none';
        bottomEl.style.display = 'none';
        return;
    } else {
        topEl.style.display = 'block';
        bottomEl.style.display = 'block';
    }
    
    topEl.textContent = formatTime(topTimeMs);
    bottomEl.textContent = formatTime(bottomTimeMs);

    if (gameActive) {
        if (isThinking) {
            topEl.classList.add('active');
            bottomEl.classList.remove('active');
        } else {
            topEl.classList.remove('active');
            bottomEl.classList.add('active');
        }
    } else {
        topEl.classList.remove('active');
        bottomEl.classList.remove('active');
    }

    topEl.classList.toggle('low-time', topTimeMs > 0 && topTimeMs <= 30000);
    bottomEl.classList.toggle('low-time', bottomTimeMs > 0 && bottomTimeMs <= 30000);
}

function formatTime(ms) {
    if (ms <= 0) return "00:00";
    
    const totalSeconds = Math.ceil(ms / 1000);
    const m = Math.floor(totalSeconds / 60);
    const s = totalSeconds % 60;
    
    if (ms <= 10000) {
        const sec = Math.floor(ms / 1000);
        const tenths = Math.floor((ms % 1000) / 100);
        return `00:${sec.toString().padStart(2, '0')}.${tenths}`;
    }
    
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

function handleTimeout(player) {
    gameActive = false;
    if (clockInterval) clearInterval(clockInterval);
    updateClockDisplays();
    
    let resStr = '';
    if (player === 'top') {
        resStr = (playerColor === 'white') ? '1-0' : '0-1';
    } else {
        resStr = (playerColor === 'white') ? '0-1' : '1-0';
    }
    
    showGameOverDialog({
        result: resStr,
        message: "Timeout"
    });
}
