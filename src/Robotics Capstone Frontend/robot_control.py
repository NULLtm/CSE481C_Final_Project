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
SERVICE_TYPE_MOVE = "interfaces/Move"

# Seconds to wait for a service response — robot moves are slow.
SERVICE_TIMEOUT = 120.0


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

    def __init__(self) -> None:
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

        self._ros = self._rp.Ros(host=ROSBRIDGE_HOST, port=ROSBRIDGE_PORT)
        log.info("RosBridge: connecting to ws://%s:%d …", ROSBRIDGE_HOST, ROSBRIDGE_PORT)

        # run() starts the event loop AND blocks until connected (or times out).
        # We call it in a daemon thread so FastAPI startup is never blocked.
        def _connect():
            try:
                self._ros.run(timeout=10.0)
                log.info("RosBridge: connected to ws://%s:%d", ROSBRIDGE_HOST, ROSBRIDGE_PORT)
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

    def __init__(self) -> None:
        self._bridge = _RosBridge()
        self._busy   = False
        self._latest_board_state: dict[int, str] = {}
        self._bridge.subscribe_board_state(self._on_board_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_busy(self) -> bool:
        return self._busy

    def reset(self) -> None:
        """Send the robot to its stow/home position via the /chess/reset service."""
        log.info("Stretch reset requested.")
        self._bridge.call_service(
            "/chess/reset",
            "std_srvs/Trigger",
            {},
        )

    def get_board_state(self) -> dict[int, str]:
        """Return the latest piece→square mapping received from the robot camera."""
        return dict(self._latest_board_state)

    def _on_board_state(self, pieces: dict[int, str]) -> None:
        self._latest_board_state = pieces

    async def execute_move(
        self,
        from_square: str,
        to_square: str,
        is_capture: bool,
        castling_rook: tuple[str, str] | None = None,
        ep_capture_sq: str | None = None,
        promo_marker_id: int | None = None,
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
        """
        if self._busy:
            log.warning("Robot busy — dropping move %s → %s", from_square, to_square)
            return
        self._busy = True
        log.info(
            "Robot executing: %s → %s  (capture=%s castle=%s ep=%s)",
            from_square, to_square, is_capture, castling_rook, ep_capture_sq,
        )
        try:
            await asyncio.to_thread(
                self._execute_sync,
                from_square, to_square, is_capture,
                castling_rook, ep_capture_sq,
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
    ) -> None:
        # 1. Remove the captured piece before moving the capturing piece.
        #    For en passant the captured pawn is on a different square.
        if is_capture:
            capture_sq = ep_capture_sq if ep_capture_sq else to_square
            self._take(capture_sq)

        # 2. Move the main piece from its source to its destination.
        self._move(from_square, to_square)

        # 3. For castling, also move the rook after the king is placed.
        if castling_rook is not None:
            rook_from, rook_to = castling_rook
            self._move(rook_from, rook_to)

    # ------------------------------------------------------------------
    # Service call helpers
    # ------------------------------------------------------------------

    def _move(self, from_sq: str, to_sq: str) -> bool:
        """Call /chess/move to pick up the piece at from_sq and place it at to_sq."""
        from_file, from_rank = _parse_square(from_sq)
        to_file,   to_rank   = _parse_square(to_sq)
        log.info("  [MOVE] %s → %s", from_sq.upper(), to_sq.upper())
        response = self._bridge.call_service(
            SERVICE_MOVE,
            SERVICE_TYPE_MOVE,
            {
                "start_file": from_file,
                "start_rank": from_rank,
                "end_file":   to_file,
                "end_rank":   to_rank,
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
        log.info("  [TAKE] clearing %s", square.upper())
        # chess_driver's take_callback reads end_file / end_rank for the target square.
        response = self._bridge.call_service(
            SERVICE_TAKE,
            SERVICE_TYPE_MOVE,
            {
                "start_file": file,
                "start_rank": rank,
                "end_file":   file,
                "end_rank":   rank,
            },
        )
        ok = bool(response and response.get("success", False))
        if not ok:
            log.error("  [TAKE] failed: %s", response)
        else:
            log.info("  [TAKE] OK — %s", response.get("message", ""))
        return ok
