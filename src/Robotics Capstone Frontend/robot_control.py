"""
Accessible Robotic Chess Player — Robot Manipulation Controller

Connects to chess_driver.py on the Stretch robot via the rosbridge WebSocket
server running on the robot at ws://ROSBRIDGE_HOST:9090.

All kinematics are handled by chess_driver.py on the robot side.
This module only translates chess move events into /chess/move and /chess/take
ROS service calls over the rosbridge connection.

Robot setup (run on the Stretch before starting the frontend):
    ros2 launch rosbridge_server rosbridge_websocket_launch.xml
    ros2 launch aaso_final_project chess_driver.launch.py
"""

import asyncio
import json
import logging
import threading

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rosbridge connection settings
# ---------------------------------------------------------------------------
ROSBRIDGE_HOST = "172.28.7.137"
ROSBRIDGE_PORT = 9090

# ---------------------------------------------------------------------------
# ROS topics and services
# ---------------------------------------------------------------------------
BOARD_STATE_TOPIC = "/chess/board_state"   # std_msgs/String (JSON from aruco node)

SERVICE_MOVE      = "/chess/move"          # interfaces/srv/Move
SERVICE_TAKE      = "/chess/take"          # interfaces/srv/Move (same type)
SERVICE_TYPE_MOVE = "interfaces/srv/Move"

# Seconds to wait for a service response — robot moves are slow.
SERVICE_TIMEOUT = 120.0


# ---------------------------------------------------------------------------
# ArUco marker ID → piece name  (matches stretch_marker_dict.yaml, IDs 1-32)
# ---------------------------------------------------------------------------
MARKER_NAMES: dict[int, str] = {
    1:  "BlackRook1",   2:  "BlackPawn1",   3:  "WhitePawn1",   4:  "WhiteRook1",
    5:  "BlackKnight1", 6:  "BlackPawn2",   7:  "WhitePawn2",   8:  "WhiteKnight1",
    9:  "BlackBishop1", 10: "BlackPawn3",   11: "WhitePawn3",   12: "WhiteBishop1",
    13: "BlackQueen1",  14: "BlackPawn4",   15: "WhitePawn4",   16: "WhiteQueen1",
    17: "BlackKing1",   18: "BlackPawn5",   19: "WhitePawn5",   20: "WhiteKing1",
    21: "BlackBishop2", 22: "BlackPawn6",   23: "WhitePawn6",   24: "WhiteBishop2",
    25: "BlackKnight2", 26: "BlackPawn7",   27: "WhitePawn7",   28: "WhiteKnight2",
    29: "BlackRook2",   30: "BlackPawn8",   31: "WhiteRook2",   32: "WhitePawn8",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_square(square: str) -> tuple[str, str]:
    """Split a UCI square string (e.g. 'e4') into (file='e', rank='4')."""
    return square[0].lower(), square[1]


# ---------------------------------------------------------------------------
# _RosBridge — roslibpy wrapper
# ---------------------------------------------------------------------------

class _RosBridge:
    """
    Thin wrapper around roslibpy that provides synchronous service calls
    and topic subscriptions over the rosbridge WebSocket.

    Degrades gracefully to log-only mode when roslibpy is not installed or
    the robot is unreachable — the rest of the game logic still runs.
    """

    def __init__(self, host, port) -> None:
        try:
            import roslibpy
            self._rp = roslibpy
        except ImportError:
            log.warning(
                "roslibpy not installed — robot will NOT move. "
                "Install with: pip install roslibpy"
            )
            self._rp = None
            self._ros = None
            return

        self._ros = self._rp.Ros(host=host, port=port)
        log.info("RosBridge: connecting to ws://%s:%d …", host, port)

        # run() starts the event loop AND blocks until connected (or times out).
        # We call it in a daemon thread so FastAPI startup is never blocked.
        def _connect():
            try:
                self._ros.run(timeout=10.0)
                log.info("RosBridge: connected to ws://%s:%d", host, port)
            except Exception as exc:
                log.warning("RosBridge: could not connect (%s) — robot moves disabled.", exc)

        threading.Thread(target=_connect, daemon=True).start()

    @property
    def connected(self) -> bool:
        return self._ros is not None and self._ros.is_connected

    def call_service(self, name: str, srv_type: str, request_dict: dict) -> dict | None:
        """
        Call a ROS service and block until the response arrives.

        Returns the response as a dict on success, or None if the connection
        is down, the call times out, or the service returns an error.
        """
        if self._rp is None:
            log.info("[log-only] %s %s", name, request_dict)
            return {"success": True, "message": "log-only mode"}

        if not self.connected:
            log.warning("RosBridge not connected — dropping service call: %s", name)
            return None

        result = [None]
        error  = [None]
        done   = threading.Event()

        def _on_success(response):
            result[0] = response
            done.set()

        def _on_error(err):
            error[0] = err
            done.set()

        service = self._rp.Service(self._ros, name, srv_type)
        service.call(self._rp.ServiceRequest(request_dict), _on_success, _on_error)

        if not done.wait(timeout=SERVICE_TIMEOUT):
            log.warning("Service call timed out after %.0fs: %s", SERVICE_TIMEOUT, name)
            return None

        if error[0] is not None:
            log.error("Service error (%s): %s", name, error[0])
            return None

        return result[0]

    def subscribe_board_state(self, callback) -> None:
        """
        Subscribe to BOARD_STATE_TOPIC.
        `callback` receives a dict {piece_id: square_label} on each update.
        """
        if self._ros is None:
            return

        sub = self._rp.Topic(self._ros, BOARD_STATE_TOPIC, "std_msgs/String")

        def _cb(msg: dict) -> None:
            try:
                payload = json.loads(msg.get("data", "{}"))
                pieces = {int(k): v for k, v in payload.get("pieces", {}).items()}
                callback(pieces)
            except (ValueError, KeyError, TypeError):
                pass

        sub.subscribe(_cb)


# ---------------------------------------------------------------------------
# RobotController — public API used by main.py
# ---------------------------------------------------------------------------

class RobotController:
    """
    Translates chess move events into /chess/move and /chess/take service
    calls on the robot, routed through the rosbridge WebSocket.

    chess_driver.py on the robot owns all kinematic logic; this class only
    decides which services to call and in what order.
    """

    def __init__(self, host, port) -> None:
        self._bridge = _RosBridge(host, port)
        self._busy   = False
        self._latest_board_state: dict[int, str] = {}
        self._bridge.subscribe_board_state(self._on_board_state)


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_busy(self) -> bool:
        return self._busy

    def get_board_state(self) -> dict[int, str]:
        """Return the latest piece→square mapping received from the robot camera."""
        return dict(self._latest_board_state)

    def _on_board_state(self, pieces: dict[int, str]) -> None:
        self._latest_board_state = pieces

    def _piece_name_at(self, square: str) -> str:
        """
        Return the ArUco marker name of the piece currently at `square`,
        e.g. 'WhitePawn3'.  Falls back to 'unknown' if the camera hasn't
        seen a piece there or the board state hasn't been received yet.
        """
        for marker_id, sq in self._latest_board_state.items():
            if sq == square:
                return MARKER_NAMES.get(marker_id, "unknown")
        return "unknown"

    async def execute_move(
        self,
        from_square: str,
        to_square: str,
        is_capture: bool,
        castling_rook: tuple[str, str] | None = None,
        ep_capture_sq: str | None = None,
        promo_marker_id: int | None = None,
        piece_name: str = "",
    ) -> None:
        """
        Orchestrate the physical move sequence without blocking the asyncio
        event loop.  A concurrent call while busy is silently dropped.

        Args:
            from_square:   UCI origin square  (e.g. 'e2').
            to_square:     UCI destination square (e.g. 'e4').
            is_capture:    True for captures and en passant.
            castling_rook: (rook_from, rook_to) for castling, else None.
            ep_capture_sq: Square of the pawn removed by en passant, else None.
            promo_marker_id: Ignored — chess_driver handles promotions as a
                             regular move; piece swap is done manually.
            piece_name:    Full ArUco marker name of the moving piece from the
                           UI identity tracker (e.g. 'WhitePawn3').
        """
        if self._busy:
            log.warning("Robot busy — dropping move %s → %s", from_square, to_square)
            return
        self._busy = True
        log.info(
            "Robot executing: %s → %s  (capture=%s castle=%s ep=%s piece=%s)",
            from_square, to_square, is_capture, castling_rook, ep_capture_sq, piece_name,
        )
        try:
            await asyncio.to_thread(
                self._execute_sync,
                from_square, to_square, is_capture,
                castling_rook, ep_capture_sq, piece_name,
            )
        except Exception:
            log.exception("Unhandled error during move %s → %s", from_square, to_square)
        finally:
            self._busy = False
            log.info("Robot idle after: %s → %s", from_square, to_square)

    # ------------------------------------------------------------------
    # Synchronous sequence (runs inside asyncio.to_thread)
    # ------------------------------------------------------------------

    def _execute_sync(
        self,
        from_square: str,
        to_square: str,
        is_capture: bool,
        castling_rook: tuple[str, str] | None,
        ep_capture_sq: str | None,
        piece_name: str = "",
    ) -> None:
        # Resolve the captured piece name BEFORE calling _take so the camera
        # state is still intact (camera still sees the piece on the board).
        captured_piece_name = "unknown"
        if is_capture:
            capture_sq = ep_capture_sq if ep_capture_sq else to_square
            captured_piece_name = self._piece_name_at(capture_sq)
            self._take(capture_sq)

        # Move the main piece; pass both piece_start and piece_end so the
        # robot driver has full context (piece_end is "unknown" for non-captures).
        self._move(from_square, to_square, piece_name, captured_piece_name)

        # For castling, also move the rook (camera lookup for rook identity).
        if castling_rook is not None:
            rook_from, rook_to = castling_rook
            self._move(rook_from, rook_to)

    # ------------------------------------------------------------------
    # Service call helpers
    # ------------------------------------------------------------------

    def _move(
        self,
        from_sq: str,
        to_sq: str,
        piece_name: str = "",
        captured_piece: str = "unknown",
    ) -> bool:
        """Call /chess/move to pick up the piece at from_sq and place it at to_sq.

        piece_name    → piece_start field: full marker name from UI identity
                        tracker (e.g. 'WhitePawn3'); falls back to camera if empty.
        captured_piece → piece_end field: marker name of the piece that was at
                        to_sq before the move ('unknown' for non-captures).
        """
        from_file, from_rank = _parse_square(from_sq)
        to_file,   to_rank   = _parse_square(to_sq)
        piece_start = piece_name if piece_name else self._piece_name_at(from_sq)
        log.info(
            "  [MOVE] %s → %s  (piece_start=%s piece_end=%s)",
            from_sq.upper(), to_sq.upper(), piece_start, captured_piece,
        )
        response = self._bridge.call_service(
            SERVICE_MOVE,
            SERVICE_TYPE_MOVE,
            {
                "start_file":  from_file,
                "start_rank":  from_rank,
                "end_file":    to_file,
                "end_rank":    to_rank,
                "piece_start": piece_start,
                "piece_end":   captured_piece,
            },
        )
        ok = bool(response and response.get("success", False))
        if not ok:
            log.error("  [MOVE] failed: %s", response)
        else:
            log.info("  [MOVE] OK — %s", response.get("message", ""))
        return ok

    def _take(self, square: str) -> bool:
        """Call /chess/take to pick up and discard the piece at square."""
        file, rank = _parse_square(square)
        piece = self._piece_name_at(square)
        log.info("  [TAKE] clearing %s  (piece_start=%s)", square.upper(), piece)
        # chess_driver's take_callback reads end_file / end_rank for the target square.
        response = self._bridge.call_service(
            SERVICE_TAKE,
            SERVICE_TYPE_MOVE,
            {
                "start_file":  file,
                "start_rank":  rank,
                "end_file":    file,
                "end_rank":    rank,
                "piece_start": piece,
                "piece_end":   "unknown",
            },
        )
        ok = bool(response and response.get("success", False))
        if not ok:
            log.error("  [TAKE] failed: %s", response)
        else:
            log.info("  [TAKE] OK — %s", response.get("message", ""))
        return ok
