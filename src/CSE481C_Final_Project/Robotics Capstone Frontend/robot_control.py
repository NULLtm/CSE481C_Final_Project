"""
Accessible Robotic Chess Player — Robot Manipulation Controller
Abstracted workspace controller; ROS 2 / stretch_body intentionally excluded
to prevent environment crashes. All hardware calls are stubbed with verbose logging.
"""

import math
import time

# ---------------------------------------------------------------------------
# Board-to-Workspace Calibration Constants
# (Tune these values to match the physical setup once hardware is available.)
# ---------------------------------------------------------------------------

# Robot base frame origin is assumed to be at the front-left corner of the board.
BOARD_ORIGIN_X_M  = 0.50   # metres forward from robot base to a1 square centre
BOARD_ORIGIN_Y_M  = -0.30  # metres lateral from robot centre-line to a1 square centre
SQUARE_SIZE_M     = 0.057  # standard FIDE square = 57 mm

ARM_TRAVEL_HEIGHT_M  = 0.35   # safe Z height while traversing between squares
ARM_PICK_HEIGHT_M    = 0.10   # Z height to close gripper around a piece
ARM_PLACE_HEIGHT_M   = 0.12   # Z height to open gripper when placing

GRIPPER_OPEN_POS     = 0.08   # metres (gripper aperture)
GRIPPER_CLOSED_POS   = 0.01

# Designated "graveyard" coordinate where captured pieces are deposited.
GRAVEYARD_X_M = 0.25
GRAVEYARD_Y_M = 0.55
GRAVEYARD_Z_M = 0.20

# Simulated motion delays (seconds) — remove when running real hardware.
SIM_DELAY_TRAVERSE = 0.3
SIM_DELAY_VERTICAL = 0.15
SIM_DELAY_GRIPPER  = 0.10


# ---------------------------------------------------------------------------
# Helper Utilities
# ---------------------------------------------------------------------------

def _square_to_coords(square: str) -> tuple[float, float, float]:
    """
    Convert a UCI square name (e.g. 'e4') to robot workspace (X, Y, Z) coords.
    File a=0 … h=7; Rank 1=0 … 8=7.
    """
    file_idx = ord(square[0].lower()) - ord('a')  # 0-7
    rank_idx = int(square[1]) - 1                 # 0-7

    x = BOARD_ORIGIN_X_M + rank_idx * SQUARE_SIZE_M
    y = BOARD_ORIGIN_Y_M + file_idx * SQUARE_SIZE_M
    z = ARM_TRAVEL_HEIGHT_M

    return round(x, 4), round(y, 4), round(z, 4)


def _sim_sleep(seconds: float, label: str = "") -> None:
    """Simulates hardware motion delay with an optional progress label."""
    tag = f" [{label}]" if label else ""
    print(f"        ~ simulating{tag}: {seconds}s motion delay")
    time.sleep(seconds)


# ---------------------------------------------------------------------------
# Robot Controller
# ---------------------------------------------------------------------------

class RobotController:
    """
    High-level chess move executor for the Hello Robot Stretch.

    Hardware interfaces (ROS 2 topics, stretch_body API calls) are replaced
    with print statements and sleep-based simulation. Replace the `_stub_*`
    methods with real driver calls once the environment is available.
    """

    def __init__(self) -> None:
        print("[RobotController] Initialised in STUB mode — no hardware connected.")
        self._gripper_state = "open"
        self._arm_position = (0.0, 0.0, ARM_TRAVEL_HEIGHT_M)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def execute_move(self, from_square: str, to_square: str, is_capture: bool) -> None:
        """
        Orchestrate the full physical move sequence.

        Args:
            from_square: UCI origin square, e.g. 'e2'.
            to_square:   UCI destination square, e.g. 'e4'.
            is_capture:  True if the move captures an opponent piece.
        """
        print("\n" + "=" * 60)
        print(f"[RobotController] EXECUTE MOVE: {from_square.upper()} → {to_square.upper()}"
              f"  (capture={is_capture})")
        print("=" * 60)

        # Step 1 — calculate workspace targets.
        from_coords = _square_to_coords(from_square)
        to_coords   = _square_to_coords(to_square)
        self._log_coords("FROM", from_square, from_coords)
        self._log_coords("TO",   to_square,   to_coords)

        # Step 2 — if capture, remove opponent piece first.
        if is_capture:
            self._execute_capture_sequence(to_square, to_coords)

        # Step 3 — pick up the moving piece from its origin square.
        self._pick_piece(from_square, from_coords)

        # Step 4 — place the piece on the destination square.
        self._place_piece(to_square, to_coords)

        print(f"[RobotController] Move complete: {from_square.upper()} → {to_square.upper()}\n")

    # ------------------------------------------------------------------ #
    # Sub-sequences                                                        #
    # ------------------------------------------------------------------ #

    def _execute_capture_sequence(self, square: str, coords: tuple) -> None:
        """Navigate to the occupied square, pick up the captured piece, dispose it."""
        print(f"\n  [CAPTURE] Removing opponent piece from {square.upper()}")

        print(f"  [CAPTURE] 1. Traversing to {square.upper()} at {coords}")
        self._stub_move_arm(coords[0], coords[1], ARM_TRAVEL_HEIGHT_M)

        print(f"  [CAPTURE] 2. Lowering arm to pick height Z={ARM_PICK_HEIGHT_M}m")
        self._stub_move_arm(coords[0], coords[1], ARM_PICK_HEIGHT_M)

        print(f"  [CAPTURE] 3. Closing gripper around captured piece")
        self._stub_set_gripper(GRIPPER_CLOSED_POS)

        print(f"  [CAPTURE] 4. Lifting to travel height")
        self._stub_move_arm(coords[0], coords[1], ARM_TRAVEL_HEIGHT_M)

        print(f"  [CAPTURE] 5. Traversing to graveyard "
              f"({GRAVEYARD_X_M}, {GRAVEYARD_Y_M}, {GRAVEYARD_Z_M})")
        self._stub_move_arm(GRAVEYARD_X_M, GRAVEYARD_Y_M, ARM_TRAVEL_HEIGHT_M)

        print(f"  [CAPTURE] 6. Lowering to graveyard deposit height Z={GRAVEYARD_Z_M}m")
        self._stub_move_arm(GRAVEYARD_X_M, GRAVEYARD_Y_M, GRAVEYARD_Z_M)

        print(f"  [CAPTURE] 7. Opening gripper — piece released into graveyard")
        self._stub_set_gripper(GRIPPER_OPEN_POS)

        print(f"  [CAPTURE] 8. Retracting arm to travel height")
        self._stub_move_arm(GRAVEYARD_X_M, GRAVEYARD_Y_M, ARM_TRAVEL_HEIGHT_M)

        print(f"  [CAPTURE] Capture sequence complete.\n")

    def _pick_piece(self, square: str, coords: tuple) -> None:
        """Lower to origin square and grip the moving piece."""
        print(f"  [PICK] Acquiring piece on {square.upper()}")

        print(f"  [PICK] 1. Traversing above {square.upper()} — XY=({coords[0]}, {coords[1]})")
        self._stub_move_arm(coords[0], coords[1], ARM_TRAVEL_HEIGHT_M)

        print(f"  [PICK] 2. Lowering to pick height Z={ARM_PICK_HEIGHT_M}m")
        self._stub_move_arm(coords[0], coords[1], ARM_PICK_HEIGHT_M)

        print(f"  [PICK] 3. Closing gripper — grasping piece")
        self._stub_set_gripper(GRIPPER_CLOSED_POS)

        print(f"  [PICK] 4. Lifting piece to travel height Z={ARM_TRAVEL_HEIGHT_M}m")
        self._stub_move_arm(coords[0], coords[1], ARM_TRAVEL_HEIGHT_M)

        print(f"  [PICK] Pick sequence complete.\n")

    def _place_piece(self, square: str, coords: tuple) -> None:
        """Traverse to destination square and release the piece."""
        print(f"  [PLACE] Placing piece on {square.upper()}")

        print(f"  [PLACE] 1. Traversing above {square.upper()} — XY=({coords[0]}, {coords[1]})")
        self._stub_move_arm(coords[0], coords[1], ARM_TRAVEL_HEIGHT_M)

        print(f"  [PLACE] 2. Lowering to place height Z={ARM_PLACE_HEIGHT_M}m")
        self._stub_move_arm(coords[0], coords[1], ARM_PLACE_HEIGHT_M)

        print(f"  [PLACE] 3. Opening gripper — releasing piece")
        self._stub_set_gripper(GRIPPER_OPEN_POS)

        print(f"  [PLACE] 4. Retracting arm to travel height Z={ARM_TRAVEL_HEIGHT_M}m")
        self._stub_move_arm(coords[0], coords[1], ARM_TRAVEL_HEIGHT_M)

        print(f"  [PLACE] Place sequence complete.\n")

    # ------------------------------------------------------------------ #
    # Hardware Stubs                                                       #
    # ------------------------------------------------------------------ #

    def _stub_move_arm(self, x: float, y: float, z: float) -> None:
        """
        STUB: Move arm end-effector to (x, y, z) in the robot base frame.
        Replace with:  robot.arm.move_to(joint_position) via stretch_body,
                       or publish to /stretch/joint_states via ROS 2.
        """
        dist = math.sqrt(
            (x - self._arm_position[0]) ** 2 +
            (y - self._arm_position[1]) ** 2 +
            (z - self._arm_position[2]) ** 2
        )
        delay = max(SIM_DELAY_TRAVERSE, dist * 0.8)  # scale delay with distance
        _sim_sleep(delay, f"arm → ({x}, {y}, {z})")
        self._arm_position = (x, y, z)
        print(f"        ✓ Arm at ({x}, {y}, {z}) m")

    def _stub_set_gripper(self, position: float) -> None:
        """
        STUB: Set gripper aperture.
        Replace with:  robot.end_of_arm.move_to('stretch_gripper', position)
                       or publish to /stretch/gripper_position via ROS 2.
        """
        state = "OPEN" if position >= GRIPPER_OPEN_POS else "CLOSED"
        _sim_sleep(SIM_DELAY_GRIPPER, f"gripper → {state}")
        self._gripper_state = state
        print(f"        ✓ Gripper {state} (aperture={position}m)")

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _log_coords(label: str, square: str, coords: tuple) -> None:
        print(f"  [{label}] square={square.upper()}  "
              f"X={coords[0]}m  Y={coords[1]}m  Z={coords[2]}m")
