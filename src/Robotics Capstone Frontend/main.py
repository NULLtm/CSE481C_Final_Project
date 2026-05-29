"""
Accessible Robotic Chess Player — Backend & WebSocket Server
FastAPI core: validates moves via python-chess, broadcasts FEN, triggers robot.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import List

import chess
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

from robot_control import RobotController

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

    def __init__(self) -> None:
        self.board = chess.Board()

    @property
    def fen(self) -> str:
        return self.board.fen()

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

manager = ConnectionManager()
game = GameState()
robot = RobotController()


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


@app.post("/reset")
async def reset_game():
    """Reset the board to the starting position and notify all clients."""
    game.board.reset()
    await manager.broadcast({"fen": game.fen, "event": "reset"})
    log.info("Game reset.")
    return {"status": "ok", "fen": game.fen}


# ---------------------------------------------------------------------------
# WebSocket Endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)

    # Send the current board state immediately on connection.
    await ws.send_text(json.dumps({"fen": game.fen, "event": "sync"}))

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

            from_sq = data.get("from", "")
            to_sq = data.get("to", "")

            if not from_sq or not to_sq:
                await ws.send_text(json.dumps({"fen": game.fen, "event": "error",
                                               "detail": "Missing 'from' or 'to' fields"}))
                continue

            uci = f"{from_sq}{to_sq}"

            # Handle pawn promotion — default to queen if client sent no choice.
            if len(data.get("promotion", "")) == 1:
                uci += data["promotion"]

            # --- Detect special move types on the PRE-PUSH board state ---
            castling_rook = ep_capture_sq = promo_marker_id = None
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
                    if move_obj.promotion is not None:
                        promo_marker_id = 49 if game.board.turn == chess.WHITE else 50
            except ValueError:
                pass  # apply_move will reject malformed UCI below

            move = game.apply_move(uci)

            if move is not None:
                # Broadcast the updated board to all clients immediately;
                # the robot move runs concurrently and never delays the UI.
                await manager.broadcast({"fen": game.fen, "event": "move", "uci": uci})

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

                if robot.is_busy:
                    await manager.broadcast({
                        "fen": game.fen, "event": "robot_busy", "uci": uci,
                    })
                else:
                    async def _run_and_notify(
                        fsq=from_sq, tsq=to_sq,
                        cap=game.last_move_was_capture(),
                        cr=castling_rook, ep=ep_capture_sq, pm=promo_marker_id,
                    ):
                        await robot.execute_move(
                            from_square=fsq, to_square=tsq, is_capture=cap,
                            castling_rook=cr, ep_capture_sq=ep,
                            promo_marker_id=pm,
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
