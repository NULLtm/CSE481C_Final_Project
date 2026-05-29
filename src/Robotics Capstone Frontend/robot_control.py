"""
Accessible Robotic Chess Player — Robot Manipulation Controller

Connects to the Hello Robot Stretch via the rosbridge WebSocket server
running on the robot at ws://172.28.7.137:9090.

Requirements:
    pip install roslibpy

Robot setup (run on the Stretch before starting the frontend):
    ros2 launch rosbridge_server rosbridge_websocket_launch.xml
    ros2 launch stretch_core stretch_driver.launch.py

Kinematic model
---------------
The robot stands ALONGSIDE the board, arm facing perpendicular to the rank axis:

    +--- board ---+
    |  a b c d e f g h  |  <- files (a = near robot, h = far)
    |  1 2 3 4 5 6 7 8  |  <- ranks sweep via wrist yaw
    +--- board ---+
         ^
      [Stretch]

  joint_lift      → vertical height (z)
  joint_arm_l0    → arm extension   sweeps files a→h
  joint_wrist_yaw → wrist rotation  sweeps ranks 1→8
  joint_gripper_* → aperture

All CALIBRATION constants are in the section below.  Use the Stretch jogging
tools (e.g. stretch_robot_home.py or keyboard_teleop) to find correct values
for your physical setup, then update the constants here.
"""

import asyncio
import json
import logging
import threading
import time

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rosbridge connection
# ---------------------------------------------------------------------------
ROSBRIDGE_HOST = "172.28.7.137"
ROSBRIDGE_PORT = 9090

# ---------------------------------------------------------------------------
# Stretch joint names  (stretch_ros2 defaults)
# ---------------------------------------------------------------------------
JOINT_LIFT      = "joint_lift"
JOINT_ARM       = "joint_arm_l0"      # outermost telescoping stage
JOINT_WRIST_YAW = "joint_wrist_yaw"
JOINT_GRIP_L    = "joint_gripper_finger_left"
JOINT_GRIP_R    = "joint_gripper_finger_right"

# ROS 2 topics exposed by stretch_ros2 + rosbridge
TRAJ_TOPIC         = "/stretch_controller/command"   # trajectory_msgs/JointTrajectory
STATE_TOPIC        = "/stretch/joint_states"          # sensor_msgs/JointState
BOARD_STATE_TOPIC  = "/chess/board_state"             # std_msgs/String (JSON from aruco_node)

# ---------------------------------------------------------------------------
# CALIBRATION — measure or jog to find values for your physical setup
# ---------------------------------------------------------------------------

# Board square size
SQUARE_M = 0.070          # 57 mm FIDE standard

# Arm extension (metres): arm sweeps across files a (col 0) → h (col 7).
#   ARM_FILE_A + 7 * SQUARE_M must be ≤ Stretch max extension (~0.52 m).
#   Start with ARM_FILE_A ≈ 0.12 so ARM_FILE_H ≈ 0.52 m.
ARM_FILE_A = 0.12         # [CALIBRATE] extension when pointing at file a

# Wrist yaw (radians): sweeps ranks 1 (row 0) → 8 (row 7).
#   Total angular span ≈ atan2(7 * SQUARE_M, arm_midpoint).
#   At arm_midpoint ≈ 0.32 m: total ≈ 0.89 rad → WRIST_PER_RANK ≈ 0.127.
WRIST_RANK1    = -0.44    # [CALIBRATE] yaw angle aligned with rank 1
WRIST_PER_RANK =  0.127   # [CALIBRATE] radians per rank step

# Lift heights (metres)
LIFT_TRAVEL = 0.80        # clearance height for arm transit between squares
LIFT_PICK   = 0.50        # height to close gripper around a piece  [CALIBRATE]
LIFT_PLACE  = 0.52        # height to open gripper when placing     [CALIBRATE]

# Gripper joint positions (Stretch finger joints use radians)
GRIP_OPEN   =  0.165      # fully open
GRIP_CLOSED = -0.10       # gripping a chess piece  [CALIBRATE for piece diameter]

# Joint-settling tolerances and wait caps
TOL_M     = 0.03          # metres  — lift and arm
TOL_RAD   = 0.05          # radians — wrist and gripper
WAIT_ARM  = 10.0          # seconds before giving up on arm/wrist/lift move
WAIT_GRIP =  3.0          # seconds before giving up on gripper move

# ---------------------------------------------------------------------------
# Graveyard — captured pieces deposited in a 2-row × 8-col grid
# ---------------------------------------------------------------------------
GRAVE_ARM_BASE   = 0.10   # arm extension (retracted, away from the board)
GRAVE_WRIST_BASE = -0.80  # wrist yaw pointing toward the graveyard zone
GRAVE_LIFT_DROP  =  0.55  # height to release a captured piece
GRAVE_SLOT_STEP  =  0.07  # spacing between graveyard slots (metres per slot)

# ---------------------------------------------------------------------------
# Promotion staging areas (IDs 49 = WhitePromo, 50 = BlackPromo)
# One promoted piece of each type waits at each promo area before a game.
# The robot picks from here after depositing the promoting pawn.
# ---------------------------------------------------------------------------
PROMO_JOINTS: dict[int, dict] = {
    49: {   # WhitePromo — near white's back-rank corners
        JOINT_LIFT:      LIFT_TRAVEL,
        JOINT_ARM:       0.08,    # [CALIBRATE]
        JOINT_WRIST_YAW: 0.60,    # [CALIBRATE]
    },
    50: {   # BlackPromo — near black's back-rank corners
        JOINT_LIFT:      LIFT_TRAVEL,
        JOINT_ARM:       0.08,    # [CALIBRATE]
        JOINT_WRIST_YAW: -0.80,   # [CALIBRATE]
    },
}

VERIFY_TIMEOUT = 5.0   # seconds to wait for camera confirmation after a place


# ---------------------------------------------------------------------------
# _RosBridge — low-level roslibpy wrapper
# ---------------------------------------------------------------------------

class _RosBridge:
    """
    Manages the roslibpy WebSocket client.

    Gracefully degrades to log-only mode when roslibpy is not installed or
    the rosbridge server is unreachable — the rest of the game logic still
    runs and the board state is still updated for the UI.
    """

    def __init__(self) -> None:
        try:
            import roslibpy
            self._rp = roslibpy
        except ImportError:
            log.warning(
                "roslibpy not installed — robot will NOT move.  "
                "Install with: pip install roslibpy"
            )
            self._rp = None
            self._ros = None
            self._pub = None
            return

        self._ros = self._rp.Ros(host=ROSBRIDGE_HOST, port=ROSBRIDGE_PORT)
        self._ros.run_in_thread()   # background WebSocket thread; auto-reconnects
        log.info("RosBridge: connecting to ws://%s:%d …", ROSBRIDGE_HOST, ROSBRIDGE_PORT)

        self._pub = self._rp.Topic(
            self._ros, TRAJ_TOPIC, "trajectory_msgs/JointTrajectory"
        )

    @property
    def connected(self) -> bool:
        return self._ros is not None and self._ros.is_connected

    def publish_trajectory(self, joints: dict[str, float],
                           duration_sec: float = 5.0) -> None:
        """
        Publish a single-waypoint JointTrajectory to move to `joints`
        within `duration_sec` seconds.
        """
        if self._pub is None:
            log.info("[log-only] JointTrajectory → %s  (%.1f s)", joints, duration_sec)
            return
        if not self.connected:
            log.warning("RosBridge not connected — trajectory dropped.")
            return

        sec     = int(duration_sec)
        nanosec = int((duration_sec - sec) * 1_000_000_000)
        msg = {
            "joint_names": list(joints.keys()),
            "points": [{
                "positions":     list(joints.values()),
                "velocities":    [],
                "accelerations": [],
                "effort":        [],
                "time_from_start": {"sec": sec, "nanosec": nanosec},
            }],
        }
        self._pub.publish(self._rp.Message(msg))
        log.debug("JointTrajectory published → %s", joints)

    def wait_for_joints(self, targets: dict[str, float],
                        timeout: float = WAIT_ARM) -> bool:
        """
        Subscribe to STATE_TOPIC and block until every joint in `targets` is
        within tolerance or `timeout` seconds elapses.

        Returns True if all targets were reached, False if timed out.
        """
        if self._ros is None:
            # Log-only mode: simulate settling time proportional to timeout.
            time.sleep(min(timeout * 0.25, 2.0))
            return True
        if not self.connected:
            log.warning("RosBridge not connected — cannot wait for joint state.")
            return False

        reached  = [False]
        done_evt = threading.Event()

        def _on_state(msg: dict) -> None:
            if done_evt.is_set():
                return
            names     = msg.get("name", [])
            positions = msg.get("position", [])
            state     = dict(zip(names, positions))
            for jname, jtarget in targets.items():
                tol = TOL_RAD if jname in (
                    JOINT_WRIST_YAW, JOINT_GRIP_L, JOINT_GRIP_R
                ) else TOL_M
                if abs(state.get(jname, float("inf")) - jtarget) > tol:
                    return   # at least one joint still off-target
            reached[0] = True
            done_evt.set()

        sub = self._rp.Topic(self._ros, STATE_TOPIC, "sensor_msgs/JointState")
        sub.subscribe(_on_state)
        done_evt.wait(timeout=timeout)
        sub.unsubscribe()

        if not reached[0]:
            log.warning("Joint targets not reached in %.1fs — proceeding.", timeout)
        return reached[0]

    def subscribe_board_state(self, callback) -> None:
        """
        Subscribe to BOARD_STATE_TOPIC (published by aruco_node.py on the robot).
        `callback` receives a dict {piece_id: square_label} on each camera update.
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
# Kinematics helpers
# ---------------------------------------------------------------------------

def _square_to_joints(square: str, lift: float) -> dict[str, float]:
    """
    Convert a chess square name (e.g. 'e4') to Stretch joint targets.
    Uses the alongside-the-board kinematic model (see module docstring).
    """
    file_idx = ord(square[0].lower()) - ord("a")   # 0 = file a, 7 = file h
    rank_idx = int(square[1]) - 1                   # 0 = rank 1, 7 = rank 8
    return {
        JOINT_LIFT:      lift,
        JOINT_ARM:       ARM_FILE_A + file_idx * SQUARE_M,
        JOINT_WRIST_YAW: WRIST_RANK1 + rank_idx * WRIST_PER_RANK,
    }


def _graveyard_joints(slot: int, lift: float) -> dict[str, float]:
    """
    Return joint targets for the `slot`-th graveyard position.
    Slots fill left-to-right in the first row, then continue in a second row.
    """
    col = slot % 8
    row = slot // 8
    return {
        JOINT_LIFT:      lift,
        JOINT_ARM:       GRAVE_ARM_BASE   + col * GRAVE_SLOT_STEP,
        JOINT_WRIST_YAW: GRAVE_WRIST_BASE + row * GRAVE_SLOT_STEP,
    }


# ---------------------------------------------------------------------------
# RobotController — public API
# ---------------------------------------------------------------------------

class RobotController:
    """
    High-level chess move executor for the Hello Robot Stretch.

    execute_move() is an async coroutine that runs the full physical move
    sequence in a background thread so it never blocks the asyncio event loop.
    The is_busy property lets callers gate concurrent move attempts.
    """

    def __init__(self) -> None:
        self._bridge              = _RosBridge()
        self._busy                = False
        self._graveyard_slot      = 0
        self._latest_board_state: dict[int, str] = {}
        self._bridge.subscribe_board_state(self._on_board_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_busy(self) -> bool:
        return self._busy

    def get_board_state(self) -> dict[int, str]:
        """Return the latest piece→square mapping received from aruco_node."""
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
        Orchestrate the full physical move sequence without blocking the
        asyncio event loop.  A concurrent call while busy is silently dropped.

        Args:
            from_square:     UCI origin square.
            to_square:       UCI destination square.
            is_capture:      True for normal captures and en passant.
            castling_rook:   (rook_from, rook_to) for castling moves, else None.
            ep_capture_sq:   Square of the pawn removed by en passant, else None.
            promo_marker_id: 49 (WhitePromo) or 50 (BlackPromo) for pawn promotions.
        """
        if self._busy:
            log.warning("Robot busy — dropping move %s → %s", from_square, to_square)
            return
        self._busy = True
        log.info("Robot executing: %s → %s  (capture=%s castle=%s ep=%s promo=%s)",
                 from_square, to_square, is_capture,
                 castling_rook, ep_capture_sq, promo_marker_id)
        try:
            await asyncio.to_thread(
                self._execute_sync,
                from_square, to_square, is_capture,
                castling_rook, ep_capture_sq, promo_marker_id,
            )
        except Exception:
            log.exception("Unhandled error during move %s → %s",
                          from_square, to_square)
        finally:
            self._busy = False
            log.info("Robot idle after: %s → %s", from_square, to_square)

    # ------------------------------------------------------------------
    # Synchronous move sequence (runs in asyncio.to_thread worker)
    # ------------------------------------------------------------------

    def _execute_sync(
        self,
        from_square: str,
        to_square: str,
        is_capture: bool,
        castling_rook: tuple[str, str] | None,
        ep_capture_sq: str | None,
        promo_marker_id: int | None,
    ) -> None:
        # 1. Remove captured piece.  For en passant the captured pawn is on a
        #    different square than the destination.
        if is_capture:
            self._capture(ep_capture_sq if ep_capture_sq else to_square)

        # 2. Move the main piece.
        if promo_marker_id is not None:
            # Promotion: pawn goes to graveyard, promoted piece comes from promo area.
            self._pick(from_square)
            self._deposit_to_graveyard()
            self._pick_from_promo(promo_marker_id)
            self._place(to_square)
        else:
            self._pick(from_square)
            self._place(to_square)

        # 3. Castling: also move the rook after the king is placed.
        if castling_rook is not None:
            rook_from, rook_to = castling_rook
            self._pick(rook_from)
            self._place(rook_to)

        # 4. Post-move verification via camera (non-blocking if camera offline).
        if self._latest_board_state:
            self._verify_placement(to_square)

    def _capture(self, square: str) -> None:
        """Pick up the piece at `square` and deposit it in the graveyard."""
        log.info("  [CAPTURE] clearing %s", square.upper())
        self._pick(square)
        self._deposit_to_graveyard()

    def _deposit_to_graveyard(self) -> None:
        """Deposit whatever is currently in the gripper to the next graveyard slot."""
        grave_transit = _graveyard_joints(self._graveyard_slot, LIFT_TRAVEL)
        grave_drop    = _graveyard_joints(self._graveyard_slot, GRAVE_LIFT_DROP)
        self._move_arm(grave_transit)
        self._move_arm(grave_drop)
        self._set_gripper(GRIP_OPEN)
        self._move_arm(grave_transit)
        self._graveyard_slot += 1

    def _pick_from_promo(self, promo_marker_id: int) -> None:
        """Travel to the promotion staging area and pick up the waiting piece."""
        promo = dict(PROMO_JOINTS[promo_marker_id])
        promo_pick = {**promo, JOINT_LIFT: LIFT_PICK}
        self._move_arm(promo)
        self._move_arm(promo_pick)
        self._set_gripper(GRIP_CLOSED)
        self._move_arm(promo)

    def _verify_placement(self, square: str) -> bool:
        """
        Wait up to VERIFY_TIMEOUT seconds for any piece to appear at `square`
        in the camera board state.  Returns True if confirmed, False if timeout.
        """
        deadline = time.monotonic() + VERIFY_TIMEOUT
        while time.monotonic() < deadline:
            if square in self._latest_board_state.values():
                log.debug("Verification OK: piece detected at %s", square.upper())
                return True
            time.sleep(0.3)
        log.warning("Post-move verification: no piece seen at %s — check accuracy.",
                    square.upper())
        return False

    def _pick(self, square: str) -> None:
        """Traverse to `square`, lower, and grip the piece."""
        log.info("  [PICK]    %s", square.upper())
        self._move_arm(_square_to_joints(square, LIFT_TRAVEL))
        self._move_arm(_square_to_joints(square, LIFT_PICK))
        self._set_gripper(GRIP_CLOSED)
        self._move_arm(_square_to_joints(square, LIFT_TRAVEL))

    def _place(self, square: str) -> None:
        """Carry the held piece to `square`, lower, and release."""
        log.info("  [PLACE]   %s", square.upper())
        self._move_arm(_square_to_joints(square, LIFT_TRAVEL))
        self._move_arm(_square_to_joints(square, LIFT_PLACE))
        self._set_gripper(GRIP_OPEN)
        self._move_arm(_square_to_joints(square, LIFT_TRAVEL))

    # ------------------------------------------------------------------
    # Primitive hardware commands
    # ------------------------------------------------------------------

    def _move_arm(self, joints: dict[str, float]) -> None:
        """
        Send a JointTrajectory to the Stretch controller and wait for the
        arm to settle at the target positions before returning.
        """
        self._bridge.publish_trajectory(joints, duration_sec=WAIT_ARM)
        self._bridge.wait_for_joints(joints, timeout=WAIT_ARM)

    def _set_gripper(self, position: float) -> None:
        """Set both gripper fingers to `position` (radians) and wait."""
        joints = {JOINT_GRIP_L: position, JOINT_GRIP_R: position}
        self._bridge.publish_trajectory(joints, duration_sec=WAIT_GRIP)
        self._bridge.wait_for_joints(joints, timeout=WAIT_GRIP)
