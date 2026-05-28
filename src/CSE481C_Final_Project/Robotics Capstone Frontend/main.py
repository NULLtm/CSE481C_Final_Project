"""
Accessible Robotic Chess Player — Backend & WebSocket Server
FastAPI core: validates moves via python-chess, broadcasts FEN, triggers robot.
"""

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
    return templates.TemplateResponse("index.html", {"request": request})


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

            # Handle pawn promotion — default to queen.
            if len(data.get("promotion", "")) == 1:
                uci += data["promotion"]

            move = game.apply_move(uci)

            if move is not None:
                # Valid move: broadcast updated FEN to all clients.
                await manager.broadcast({"fen": game.fen, "event": "move", "uci": uci})

                # Trigger physical robot movement (non-blocking stub).
                robot.execute_move(
                    from_square=from_sq,
                    to_square=to_sq,
                    is_capture=game.last_move_was_capture(),
                )
            else:
                # Invalid move: echo the unchanged FEN back to the sender
                # so chessboard.js performs a visual snapback.
                await ws.send_text(json.dumps({"fen": game.fen, "event": "invalid",
                                               "uci": uci}))

    except WebSocketDisconnect:
        manager.disconnect(ws)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
