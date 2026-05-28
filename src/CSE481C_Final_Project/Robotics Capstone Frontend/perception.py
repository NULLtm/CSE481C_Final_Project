"""
Accessible Robotic Chess Player — Vision & Perception Module
OpenCV ArUco pipeline: detects board corners (IDs 33-36), applies perspective
warp to produce a top-down view, then maps piece markers (IDs 1-32) to squares.
"""

import sys
import logging
import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("OpenCV is required: pip install opencv-contrib-python")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ArUco marker IDs assigned to board corners (order matters for warp).
CORNER_IDS = {
    33: "top-left",
    34: "top-right",
    35: "bottom-right",
    36: "bottom-left",
}
CORNER_ORDER = [33, 34, 35, 36]  # must match src pts order for getPerspectiveTransform

PIECE_ID_RANGE = range(1, 33)    # IDs 1–32 represent chess pieces

# Output dimensions of the warped top-down board image.
WARP_SIZE = 800   # pixels (square image); each chess square = WARP_SIZE / 8 = 100 px

FILES = list("abcdefgh")
RANKS = list("87654321")  # rank 8 = top row in image, rank 1 = bottom row


# ---------------------------------------------------------------------------
# ArUco Setup
# ---------------------------------------------------------------------------

def _build_detector() -> cv2.aruco.ArucoDetector:
    """Construct a detector using the 6×6 250-marker dictionary."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, params)


# ---------------------------------------------------------------------------
# Geometry Helpers
# ---------------------------------------------------------------------------

def _marker_center(corners: np.ndarray) -> tuple[float, float]:
    """Return the (x, y) centroid of a single marker's four corner points."""
    pts = corners[0]  # shape (4, 2)
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    return cx, cy


def _extract_board_corners(
    ids: np.ndarray,
    corners: list,
) -> np.ndarray | None:
    """
    Locate the four board-boundary markers in the detection results and
    return their centres as a (4, 2) float32 array ordered [TL, TR, BR, BL].
    Returns None if any corner marker is missing.
    """
    id_to_centre: dict[int, tuple[float, float]] = {}
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        if marker_id in CORNER_IDS:
            id_to_centre[int(marker_id)] = _marker_center(marker_corners)

    if len(id_to_centre) < 4:
        missing = [mid for mid in CORNER_ORDER if mid not in id_to_centre]
        log.warning("Missing corner marker IDs: %s", missing)
        return None

    src_pts = np.array(
        [id_to_centre[mid] for mid in CORNER_ORDER],
        dtype=np.float32,
    )
    return src_pts


def _build_warp_transform(src_pts: np.ndarray) -> np.ndarray:
    """Compute the perspective transform matrix from board corners to a square canvas."""
    dst_pts = np.array(
        [
            [0,         0        ],
            [WARP_SIZE, 0        ],
            [WARP_SIZE, WARP_SIZE],
            [0,         WARP_SIZE],
        ],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return M


def _pixel_to_square(px: float, py: float) -> str | None:
    """
    Map a pixel coordinate in the warped (top-down) image to a chess square label.
    Returns None if the coordinate is out of bounds.
    """
    if not (0 <= px < WARP_SIZE and 0 <= py < WARP_SIZE):
        return None

    cell = WARP_SIZE / 8
    col = int(px // cell)  # 0 = file a, 7 = file h
    row = int(py // cell)  # 0 = rank 8, 7 = rank 1

    col = min(col, 7)
    row = min(row, 7)

    return FILES[col] + RANKS[row]


# ---------------------------------------------------------------------------
# Core Detection Pipeline
# ---------------------------------------------------------------------------

def detect_board_state(
    frame: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
) -> dict[int, str] | None:
    """
    Run the full perception pipeline on a single camera frame.

    Returns a mapping of {piece_id: square_label} for all detected piece markers,
    or None if the board perspective transform could not be established.
    """
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is None or len(ids) == 0:
        log.warning("No ArUco markers detected in frame.")
        return None

    # --- Establish board perspective warp ---
    src_pts = _extract_board_corners(ids, corners)
    if src_pts is None:
        return None

    M = _build_warp_transform(src_pts)

    # --- Warp the frame to top-down view (useful for visualisation / debugging) ---
    warped = cv2.warpPerspective(frame, M, (WARP_SIZE, WARP_SIZE))

    # --- Map each piece marker to a square ---
    piece_positions: dict[int, str] = {}
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        pid = int(marker_id)
        if pid not in PIECE_ID_RANGE:
            continue  # skip corner markers

        cx_orig, cy_orig = _marker_center(marker_corners)

        # Transform the marker centre point through the same perspective matrix.
        pt = np.array([[[cx_orig, cy_orig]]], dtype=np.float32)
        warped_pt = cv2.perspectiveTransform(pt, M)
        wx, wy = warped_pt[0][0]

        square = _pixel_to_square(wx, wy)
        if square:
            piece_positions[pid] = square
            log.debug("Piece ID %2d → square %s  (warped px: %.1f, %.1f)", pid, square, wx, wy)
        else:
            log.warning("Piece ID %d centre (%.1f, %.1f) is outside the warped board.", pid, wx, wy)

    return piece_positions, warped  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Mock Camera Feed Helpers
# ---------------------------------------------------------------------------

def _create_mock_frame(width: int = 1280, height: int = 720) -> np.ndarray:
    """
    Generate a synthetic camera frame with printed ArUco markers at known positions.
    Used for unit testing / simulation when no real camera is attached.
    """
    frame = np.full((height, width, 3), fill_value=200, dtype=np.uint8)  # grey background
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    marker_size = 60  # pixels

    # Corner marker positions: [TL, TR, BR, BL] matching CORNER_ORDER [33, 34, 35, 36]
    corner_positions = {
        33: (80,  80 ),   # top-left
        34: (1140, 80 ),  # top-right
        35: (1140, 580),  # bottom-right
        36: (80,  580),   # bottom-left
    }

    for mid, (ox, oy) in corner_positions.items():
        img = cv2.aruco.generateImageMarker(dictionary, mid, marker_size)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        frame[oy:oy + marker_size, ox:ox + marker_size] = img_bgr

    # Place a few piece markers (IDs 1, 2, 3) at approximate square positions.
    board_left, board_top = 80, 80
    board_right, board_bottom = 1200, 640
    sq_w = (board_right - board_left) / 8
    sq_h = (board_bottom - board_top) / 8

    piece_squares = {
        1: (0, 6),  # a2 — file 0, rank index 6 from top
        2: (4, 6),  # e2
        3: (3, 0),  # d8
    }

    for pid, (fc, rc) in piece_squares.items():
        ox = int(board_left + fc * sq_w + sq_w / 2 - marker_size / 2)
        oy = int(board_top  + rc * sq_h + sq_h / 2 - marker_size / 2)
        img = cv2.aruco.generateImageMarker(dictionary, pid, marker_size)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        oy2, ox2 = min(oy + marker_size, height), min(ox + marker_size, width)
        frame[oy:oy2, ox:ox2] = img_bgr[: oy2 - oy, : ox2 - ox]

    return frame


# ---------------------------------------------------------------------------
# Processing Loop
# ---------------------------------------------------------------------------

def run_perception_loop(use_camera: bool = False, camera_index: int = 0) -> None:
    """
    Main perception loop.

    Args:
        use_camera:   True to read from a real camera, False to use the mock frame.
        camera_index: OpenCV capture device index (ignored when use_camera=False).
    """
    detector = _build_detector()
    log.info("ArUco detector ready (DICT_6X6_250).")

    if use_camera:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            log.error("Cannot open camera index %d", camera_index)
            return
        log.info("Capturing from camera %d. Press 'q' to quit.", camera_index)
    else:
        log.info("Running in MOCK frame mode (no camera). Press 'q' to quit.")

    try:
        while True:
            if use_camera:
                ret, frame = cap.read()
                if not ret:
                    log.error("Failed to grab frame from camera.")
                    break
            else:
                frame = _create_mock_frame()

            result = detect_board_state(frame, detector)

            if result is not None:
                piece_positions, warped = result
                log.info("Detected %d piece marker(s): %s", len(piece_positions), piece_positions)

                # Draw grid overlay on the warped image for debugging.
                _draw_grid(warped)
                cv2.imshow("Warped Board (top-down)", warped)

            # Draw raw detection overlay on the original frame.
            annotated = frame.copy()
            cv2.imshow("Camera / Mock Frame", annotated)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    finally:
        if use_camera:
            cap.release()
        cv2.destroyAllWindows()
        log.info("Perception loop terminated.")


def _draw_grid(warped: np.ndarray) -> None:
    """Overlay the 8x8 square grid on the warped image in-place."""
    cell = WARP_SIZE // 8
    for i in range(1, 8):
        cv2.line(warped, (i * cell, 0), (i * cell, WARP_SIZE), (0, 200, 0), 1)
        cv2.line(warped, (0, i * cell), (WARP_SIZE, i * cell), (0, 200, 0), 1)

    # Label columns (files) and rows (ranks).
    for col, f in enumerate(FILES):
        cx = col * cell + cell // 2 - 6
        cv2.putText(warped, f, (cx, WARP_SIZE - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
    for row, r in enumerate(RANKS):
        cy = row * cell + cell // 2 + 5
        cv2.putText(warped, r, (4, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chess board perception pipeline")
    parser.add_argument("--camera", action="store_true",
                        help="Use a real camera instead of the mock frame")
    parser.add_argument("--camera-index", type=int, default=0,
                        help="OpenCV camera device index (default 0)")
    args = parser.parse_args()

    run_perception_loop(use_camera=args.camera, camera_index=args.camera_index)
