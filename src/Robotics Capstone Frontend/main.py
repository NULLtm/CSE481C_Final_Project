"""
Accessible Robotic Chess Player — Backend & WebSocket Server
FastAPI core: validates moves via python-chess, broadcasts FEN, triggers robot.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from typing import List

import chess
import uvicorn
from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from robot_control import RobotController

# ---------------------------------------------------------------------------
# Whisper transcription model (loaded once at startup)
# ---------------------------------------------------------------------------
try:
    from faster_whisper import WhisperModel
    # "base" balances speed vs accuracy on CPU.  Switch to "small" or "medium"
    # for better accuracy at the cost of more inference time.
    # Set device="cuda" and compute_type="float16" if a GPU is available.
    _whisper = WhisperModel("base", device="cpu", compute_type="int8")
    log_tmp = logging.getLogger(__name__)
    log_tmp.info("faster-whisper model loaded (base / CPU / int8)")
except Exception as _e:
    _whisper = None
    logging.getLogger(__name__).warning(
        "faster-whisper not available — /transcribe will return empty text. (%s)", _e
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection Manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Maintains the pool of active WebSocket connections."""

    def __init__(self) -> None:
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)
        log.info("Client connected. Total connections: %d", len(self.active))

    def disconnect(self, ws: WebSocket) -> None:
        self.active.remove(ws)
        log.info("Client disconnected. Total connections: %d", len(self.active))

    async def broadcast(self, message: dict) -> None:
        payload = json.dumps(message)
        for ws in list(self.active):
            try:
                await ws.send_text(payload)
            except Exception:
                self.active.remove(ws)


# ---------------------------------------------------------------------------
# Application State
# ---------------------------------------------------------------------------

class GameState:
    """Single source of truth for the chess board."""

    # ---------------------------------------------------------------------------
    # Spare promotion pieces — keyed by color ('w'/'b') then piece type ('q'/'n').
    # Ordered: first entry is used first.
    # Physical storage locations (file, rank) are defined in robot_control.py.
    # 'r' and 'b' are intentionally absent — no spare rooks/bishops available.
    # ---------------------------------------------------------------------------
    _INITIAL_PROMO_POOL: dict[str, dict[str, list[str]]] = {
        'w': {
            'q': ['WhiteQueen2', 'WhiteQueen3'],
            'n': ['WhiteKnight3', 'WhiteKnight4'],
        },
        'b': {
            'q': ['BlackQueen2', 'BlackQueen3'],
            'n': ['BlackKnight3', 'BlackKnight4'],
        },
    }

    def __init__(self) -> None:
        self.board = chess.Board()
        self._reset_promo_pool()

    def _reset_promo_pool(self) -> None:
        import copy
        self._promo_pool: dict[str, dict[str, list[str]]] = copy.deepcopy(self._INITIAL_PROMO_POOL)

    @property
    def fen(self) -> str:
        return self.board.fen()

    def available_promotions(self) -> dict[str, dict[str, bool]]:
        """Return {color: {piece_type: available}} for all colors."""
        return {
            color: {pt: len(pool) > 0 for pt, pool in by_type.items()}
            for color, by_type in self._promo_pool.items()
        }

    def can_promote(self, color: str, piece_type: str) -> bool:
        return bool(self._promo_pool.get(color, {}).get(piece_type))

    def allocate_promo_piece(self, color: str, piece_type: str) -> str | None:
        """Pop and return the next available spare piece name, or None if exhausted."""
        pool = self._promo_pool.get(color, {}).get(piece_type, [])
        if not pool:
            return None
        name = pool.pop(0)
        log.info("Allocated promotion piece: %s  (remaining %s/%s: %d)", name, color, piece_type, len(pool))
        return name

    def apply_move(self, uci: str) -> chess.Move | None:
        """
        Attempt to apply a UCI move string (e.g. 'e2e4').
        Returns the Move on success, None if illegal.
        """
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            log.warning("Malformed UCI string: %s", uci)
            return None

        if move in self.board.legal_moves:
            # Capture detection must happen before pushing the move.
            self._last_is_capture = self.board.is_capture(move)
            self.board.push(move)
            log.info("Move applied: %s — FEN: %s", uci, self.fen)
            return move

        log.warning("Illegal move attempted: %s", uci)
        return None

    def last_move_was_capture(self) -> bool:
        return getattr(self, "_last_is_capture", False)


# ---------------------------------------------------------------------------
# App Lifespan & Initialisation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Parse rosbridge address from the command line
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(
    description="Robotic Chess web server",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="Example:\n  python main.py 172.28.7.137:9090",
)
_parser.add_argument(
    "rosbridge",
    metavar="IP:PORT",
    help="Rosbridge WebSocket address of the robot, e.g. 172.28.7.137:9090",
)
_args = _parser.parse_args()

try:
    _rb_host, _rb_port_str = _args.rosbridge.rsplit(":", 1)
    _rb_port = int(_rb_port_str)
    if not _rb_host:
        raise ValueError("host is empty")
except ValueError:
    _parser.error(f"Invalid address {_args.rosbridge!r} — expected IP:PORT, e.g. 172.28.7.137:9090")

log.info("Rosbridge target: ws://%s:%d", _rb_host, _rb_port)

manager = ConnectionManager()
game = GameState()
robot = RobotController(host=_rb_host, port=_rb_port)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Chess server starting up. Initial FEN: %s", game.fen)
    yield
    log.info("Chess server shutting down.")


app = FastAPI(title="Accessible Robotic Chess", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


# ---------------------------------------------------------------------------
# HTTP Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the tablet UI."""
    return templates.TemplateResponse(request, "index.html", {"your_other_vars": "values"})


@app.get("/fen")
async def get_fen():
    """REST convenience endpoint — returns the current FEN."""
    return {"fen": game.fen}


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Receive a browser audio blob (WebM/Opus), run faster-whisper on it,
    and return the transcript as {"text": "..."}.

    faster-whisper uses ffmpeg internally to decode the audio, so ffmpeg
    must be installed on this machine:
        macOS:  brew install ffmpeg
        Ubuntu: sudo apt install ffmpeg
    """
    if _whisper is None:
        log.warning("/transcribe called but faster-whisper is not available")
        return {"text": "", "error": "faster-whisper not installed"}

    # Save the upload to a named temp file so faster-whisper can open it.
    suffix = os.path.splitext(file.filename or "audio.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        def _run():
            # transcribe() returns a lazy generator — consume it here inside
            # the worker thread so we never block the asyncio event loop.
            segments, _ = _whisper.transcribe(tmp_path, beam_size=5)
            return " ".join(seg.text for seg in segments).strip()

        text = await asyncio.to_thread(_run)
        log.info("Transcription: %r", text)
        return {"text": text}
    except Exception as exc:
        log.error("Transcription error: %s", exc)
        return {"text": "", "error": str(exc)}
    finally:
        os.unlink(tmp_path)


@app.post("/reset_stretch")
async def reset_stretch():
    """Send the Stretch robot to its home/stow position."""
    await asyncio.to_thread(robot.reset)
    return {"status": "ok"}


@app.post("/reset")
async def reset_game():
    """Reset the board to the starting position and notify all clients."""
    game.board.reset()
    game._reset_promo_pool()
    await manager.broadcast({
        "fen": game.fen, "event": "reset",
        "available_promotions": game.available_promotions(),
    })
    log.info("Game reset.")
    return {"status": "ok", "fen": game.fen}


# ---------------------------------------------------------------------------
# WebSocket Endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)

    # Send the current board state immediately on connection.
    await ws.send_text(json.dumps({
        "fen": game.fen, "event": "sync",
        "available_promotions": game.available_promotions(),
    }))

    try:
        while True:
            raw = await ws.receive_text()
            print("RECEIVED MOVE")

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                log.warning("Non-JSON message received: %s", raw)
                await ws.send_text(json.dumps({"fen": game.fen, "event": "error",
                                               "detail": "Invalid JSON"}))
                continue

            from_sq             = data.get("from", "")
            to_sq               = data.get("to", "")
            robot_move          = bool(data.get("robot", False))
            piece_name          = data.get("piece", "")           # moving piece ArUco name
            captured_piece_name = data.get("captured_piece", "")  # captured piece ArUco name
            rook_piece_name     = data.get("rook_piece", "")      # castling rook ArUco name

            if not from_sq or not to_sq:
                await ws.send_text(json.dumps({"fen": game.fen, "event": "error",
                                               "detail": "Missing 'from' or 'to' fields"}))
                continue

            uci = f"{from_sq}{to_sq}"

            # Extract promotion letter ('q','r','b','n') before appending to UCI.
            promotion_letter = data.get("promotion", "")
            if len(promotion_letter) == 1:
                uci += promotion_letter
            else:
                promotion_letter = ""

            # Capture moving color BEFORE applying the move (board.turn flips after push).
            moving_color = 'w' if game.board.turn == chess.WHITE else 'b'

            # Reject promotion early if no spare pieces are available for that type.
            if promotion_letter and not game.can_promote(moving_color, promotion_letter):
                log.warning("Promotion rejected — no spare %s/%s pieces left.", moving_color, promotion_letter)
                await ws.send_text(json.dumps({
                    "fen": game.fen, "event": "invalid", "uci": uci,
                    "available_promotions": game.available_promotions(),
                }))
                continue

            # --- Detect special move types on the PRE-PUSH board state ---
            castling_rook = ep_capture_sq = None
            try:
                move_obj = chess.Move.from_uci(uci)
                if move_obj in game.board.legal_moves:
                    if game.board.is_castling(move_obj):
                        rank = chess.square_rank(move_obj.from_square)
                        if game.board.is_kingside_castling(move_obj):
                            castling_rook = (
                                chess.square_name(chess.square(7, rank)),  # h-file
                                chess.square_name(chess.square(5, rank)),  # f-file
                            )
                        else:
                            castling_rook = (
                                chess.square_name(chess.square(0, rank)),  # a-file
                                chess.square_name(chess.square(3, rank)),  # d-file
                            )
                    if game.board.is_en_passant(move_obj):
                        ep_capture_sq = chess.square_name(chess.square(
                            chess.square_file(move_obj.to_square),
                            chess.square_rank(move_obj.from_square),
                        ))
            except ValueError:
                pass  # apply_move will reject malformed UCI below

            move = game.apply_move(uci)

            if move is not None:
                # Allocate the physical spare piece for promotions.
                promo_piece_name = game.allocate_promo_piece(moving_color, promotion_letter) if promotion_letter else ""

                # robot_executing is True when the robot will physically carry out
                # this move — the client uses this to keep the board locked until
                # robot_idle arrives.
                robot_executing = robot_move and not robot.is_busy

                # Broadcast the updated board to all clients immediately;
                # the robot move runs concurrently and never delays the UI.
                await manager.broadcast({
                    "fen": game.fen, "event": "move", "uci": uci,
                    "robot_executing": robot_executing,
                    "promo_piece_name": promo_piece_name,
                    "available_promotions": game.available_promotions(),
                })

                # Detect and broadcast game-over conditions.
                reason: str | None = None
                if game.board.is_checkmate():
                    winner = "black" if game.board.turn == chess.WHITE else "white"
                    reason = f"checkmate — {winner} wins"
                elif game.board.is_stalemate():
                    reason = "stalemate"
                elif game.board.is_insufficient_material():
                    reason = "insufficient material"
                elif game.board.is_seventyfive_moves():
                    reason = "75-move rule"
                elif game.board.is_fivefold_repetition():
                    reason = "fivefold repetition"
                if reason:
                    await manager.broadcast(
                        {"fen": game.fen, "event": "game_over", "reason": reason}
                    )

                if robot_move:
                    # Only trigger physical robot movement when the moving
                    # player has robot control enabled.
                    if robot.is_busy:
                        await manager.broadcast({
                            "fen": game.fen, "event": "robot_busy", "uci": uci,
                        })
                    else:
                        async def _run_and_notify(
                            fsq=from_sq, tsq=to_sq,
                            cap=game.last_move_was_capture(),
                            cr=castling_rook, ep=ep_capture_sq,
                            promo=promotion_letter,
                            pn=piece_name, cpn=captured_piece_name,
                            rpn=rook_piece_name,
                            ppn=promo_piece_name,
                        ):
                            await robot.execute_move(
                                from_square=fsq, to_square=tsq, is_capture=cap,
                                castling_rook=cr, ep_capture_sq=ep,
                                promotion=promo, piece_name=pn,
                                captured_piece_name=cpn, rook_piece_name=rpn,
                                promo_piece_name=ppn,
                            )
                            await manager.broadcast({"event": "robot_idle"})

                        asyncio.create_task(_run_and_notify())
            else:
                # Invalid move: echo the unchanged FEN back to the sender
                # so chessboard.js performs a visual snapback.
                await ws.send_text(json.dumps({"fen": game.fen, "event": "invalid",
                                               "uci": uci}))

    except WebSocketDisconnect:
        manager.disconnect(ws)


# ---------------------------------------------------------------------------
# Perception WebSocket  (/ws/perception)
# ---------------------------------------------------------------------------

@app.websocket("/ws/perception")
async def perception_endpoint(ws: WebSocket):
    """
    Streams the latest ArUco-detected board state to diagnostic clients at ~2 Hz.
    Payload: {"event": "board_state", "pieces": {"1": "e2", ...}}
    Connect from any browser tab or debug tool to monitor what the camera sees.
    """
    await ws.accept()
    log.info("Perception client connected.")
    try:
        while True:
            state = robot.get_board_state()
            await ws.send_text(json.dumps({
                "event": "board_state",
                "pieces": {str(k): v for k, v in state.items()},
            }))
            await asyncio.sleep(0.5)   # 2 Hz
    except WebSocketDisconnect:
        log.info("Perception client disconnected.")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    # Use HTTPS if certificates exist (required for Web Speech API on non-localhost).
    # Generate with: openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj '/CN=localhost'
    ssl_keyfile  = "key.pem"  if os.path.exists("key.pem")  else None
    ssl_certfile = "cert.pem" if os.path.exists("cert.pem") else None
    if ssl_certfile:
        log.info("Starting with HTTPS on port 8000")
    else:
        log.info("Starting with HTTP on port 8000 — voice commands only work on localhost")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=not bool(ssl_certfile),   # reload incompatible with SSL in some uvicorn versions
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
    )
