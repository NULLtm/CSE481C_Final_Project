"""
Accessible Robotic Chess Player — Vision & Perception Module
OpenCV ArUco pipeline that mirrors the marker assignments in
``CSE481C_Final_Project/config/aruco_markers.yaml``:

- IDs 1-32  : piece markers placed on top of each chess piece.
- IDs 33-40 : file fiducials FileA..FileH. Each ID is printed TWICE — once
              on the top edge and once on the bottom edge of the board —
              so the board can be flipped without changing anything.
- IDs 41-48 : rank fiducials Rank1..Rank8. Same dual-edge scheme: each ID
              appears on both the left and right edges.
- IDs 49-50 : WhitePromo / BlackPromo. Each ID is printed TWICE — once in
              each corner on that player's side of the board — giving 4
              promotion holding squares total (2 per player).

Because each file/rank ID appears twice in a top-down view, _extract_fiducials
generates TWO candidate warped-pixel correspondences per detection (one per
possible edge). cv2.findHomography with RANSAC selects the geometrically
consistent set (~50 % inlier ratio), reliably recovering the correct transform
regardless of which physical edges happen to be visible.

Promo markers are reported with their raw image-pixel position; they live
off the board and are not mapped to chess squares.
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

# Output dimensions of the warped top-down board image.
WARP_SIZE = 800           # pixels (square image); 100 px per chess square
CELL_PX = WARP_SIZE // 8  # 100 px

FILES = list("abcdefgh")
RANKS = list("87654321")  # rank 8 = top row of warped image, rank 1 = bottom row

# Piece-marker IDs to their human-readable names, mirroring aruco_markers.yaml.
PIECE_NAMES: dict[int, str] = {
    1:  "WhitePawn1",   2:  "WhiteBishop1", 3:  "BlackPawn1",   4:  "BlackPawn2",
    5:  "BlackPawn3",   6:  "BlackPawn4",   7:  "WhiteRook1",   8:  "BlackBishop1",
    9:  "BlackPawn5",  10:  "WhitePawn2",  11:  "WhiteQueen1", 12:  "BlackRook1",
    13: "BlackPawn6",  14:  "BlackBishop2", 15: "BlackQueen1", 16:  "BlackKing1",
    17: "WhitePawn3",  18:  "WhitePawn4",  19:  "WhiteKnight1", 20: "WhiteKing1",
    21: "WhiteBishop2", 22: "WhiteRook2",  23:  "BlackKnight1", 24: "BlackKnight2",
    25: "WhitePawn5",  26:  "BlackPawn7",  27:  "WhitePawn6",  28:  "BlackPawn8",
    29: "BlackRook2",  30:  "WhitePawn7",  31:  "WhitePawn8",  32:  "WhiteKnight2",
}
PIECE_IDS = set(PIECE_NAMES)

# Edge fiducials: 33-40 are file markers (one per file letter), 41-48 are rank
# markers (one per rank number). Each maps to its 0-indexed file/rank position.
FILE_MARKER_IDS: dict[int, int] = {33 + i: i for i in range(8)}  # 33->a ... 40->h
RANK_MARKER_IDS: dict[int, int] = {41 + i: i for i in range(8)}  # 41->rank1 ... 48->rank8

# Promotion holding markers (off-board); only their image position is reported.
PROMO_NAMES: dict[int, str] = {49: "WhitePromo", 50: "BlackPromo"}


def _file_warp_pts(col: int) -> tuple[tuple[float, float], tuple[float, float]]:
    """Both candidate warped-pixel positions for a file marker at column `col` (0=a..7=h).
    Returns (top-edge position, bottom-edge position); RANSAC picks the correct one."""
    x = (col + 0.5) * CELL_PX
    return ((x, -0.5 * CELL_PX), (x, WARP_SIZE + 0.5 * CELL_PX))


def _rank_warp_pts(rank_idx: int) -> tuple[tuple[float, float], tuple[float, float]]:
    """Both candidate warped-pixel positions for a rank marker at rank_idx (0=rank1..7=rank8).
    Returns (left-edge position, right-edge position); RANSAC picks the correct one."""
    # rank index 0 (rank 1) -> bottom row of board image; rank index 7 (rank 8) -> top.
    y = (7 - rank_idx + 0.5) * CELL_PX
    return ((-0.5 * CELL_PX, y), (WARP_SIZE + 0.5 * CELL_PX, y))


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


def _extract_fiducials(
    ids: np.ndarray,
    corners: list,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Build (image_pts, warp_pts) correspondences for cv2.findHomography.

    Each detected file/rank marker yields TWO candidate correspondences —
    one per possible physical edge (top/bottom for files, left/right for ranks).
    RANSAC inside _build_homography selects the geometrically consistent half
    and discards the contradictory candidates as outliers.

    Requires at least 2 unique file IDs and 2 unique rank IDs detected.
    """
    img_pts: list[tuple[float, float]] = []
    warp_pts: list[tuple[float, float]] = []
    n_file = n_rank = 0

    for marker_corners, marker_id in zip(corners, ids.flatten()):
        mid = int(marker_id)
        cx, cy = _marker_center(marker_corners)
        if mid in FILE_MARKER_IDS:
            for wp in _file_warp_pts(FILE_MARKER_IDS[mid]):
                img_pts.append((cx, cy))
                warp_pts.append(wp)
            n_file += 1
        elif mid in RANK_MARKER_IDS:
            for wp in _rank_warp_pts(RANK_MARKER_IDS[mid]):
                img_pts.append((cx, cy))
                warp_pts.append(wp)
            n_rank += 1

    if n_file < 2 or n_rank < 2:
        log.warning(
            "Insufficient board fiducials (file=%d, rank=%d). "
            "Need at least 2 of each to compute a homography.",
            n_file, n_rank,
        )
        return None

    return (
        np.asarray(img_pts, dtype=np.float32),
        np.asarray(warp_pts, dtype=np.float32),
    )


def _build_homography(
    img_pts: np.ndarray, warp_pts: np.ndarray
) -> np.ndarray | None:
    """Solve the image -> warped-board homography. Returns None on failure."""
    H, _ = cv2.findHomography(img_pts, warp_pts, method=cv2.RANSAC,
                              ransacReprojThreshold=5.0)
    if H is None:
        log.error("findHomography failed — fiducial geometry may be degenerate.")
    return H


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
) -> tuple[dict[int, str], dict[int, tuple[float, float]], np.ndarray] | None:
    """
    Run the full perception pipeline on a single camera frame.

    Returns:
        (piece_positions, promo_positions, warped) where
          - piece_positions maps piece marker ID -> square label (e.g. 17 -> "e2"),
          - promo_positions maps promo marker ID -> (image x, image y) of its centre,
          - warped is the 800x800 top-down board image.
        Returns None if the homography could not be established.
    """
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is None or len(ids) == 0:
        log.warning("No ArUco markers detected in frame.")
        return None

    fid = _extract_fiducials(ids, corners)
    if fid is None:
        return None
    img_pts, warp_pts = fid

    H = _build_homography(img_pts, warp_pts)
    if H is None:
        return None

    warped = cv2.warpPerspective(frame, H, (WARP_SIZE, WARP_SIZE))

    piece_positions: dict[int, str] = {}
    promo_positions: dict[int, tuple[float, float]] = {}

    for marker_corners, marker_id in zip(corners, ids.flatten()):
        mid = int(marker_id)
        cx, cy = _marker_center(marker_corners)

        if mid in PIECE_IDS:
            pt = np.array([[[cx, cy]]], dtype=np.float32)
            wx, wy = cv2.perspectiveTransform(pt, H)[0][0]
            square = _pixel_to_square(wx, wy)
            if square:
                piece_positions[mid] = square
                log.debug("%s (ID %d) -> %s  (warped px: %.1f, %.1f)",
                          PIECE_NAMES[mid], mid, square, wx, wy)
            else:
                log.warning("%s (ID %d) centre (%.1f, %.1f) is outside the warped board.",
                            PIECE_NAMES[mid], mid, wx, wy)
        elif mid in PROMO_NAMES:
            promo_positions[mid] = (cx, cy)
            log.debug("%s (ID %d) detected at image px (%.1f, %.1f)",
                      PROMO_NAMES[mid], mid, cx, cy)

    return piece_positions, promo_positions, warped


# ---------------------------------------------------------------------------
# Mock Camera Feed Helpers
# ---------------------------------------------------------------------------

def _create_mock_frame(width: int = 1280, height: int = 720) -> np.ndarray:
    """
    Generate a synthetic camera frame that matches the physical board layout:
    - File markers (33-40) on BOTH top and bottom edges.
    - Rank markers (41-48) on BOTH left and right edges.
    - Promo markers (49, 50) in all 4 corners — each ID appears twice,
      once per corner on that player's side (symmetric for board-flip).
    - A few piece markers on starting squares.
    """
    frame = np.full((height, width, 3), fill_value=200, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    marker_size = 40

    # Virtual board region (the 8×8 playing area only).
    board_left, board_top = 200, 100
    board_size = 480
    cell = board_size / 8
    board_right = board_left + board_size
    board_bottom = board_top + board_size

    def stamp(mid: int, cx: int, cy: int) -> None:
        img = cv2.aruco.generateImageMarker(dictionary, mid, marker_size)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        x0 = max(cx - marker_size // 2, 0)
        y0 = max(cy - marker_size // 2, 0)
        x1 = min(x0 + marker_size, width)
        y1 = min(y0 + marker_size, height)
        frame[y0:y1, x0:x1] = img_bgr[: y1 - y0, : x1 - x0]

    # File markers on the TOP and BOTTOM edges (same ID on both sides).
    for mid, col in FILE_MARKER_IDS.items():
        fx = int(board_left + (col + 0.5) * cell)
        stamp(mid, fx, int(board_top - cell * 0.5))    # top edge
        stamp(mid, fx, int(board_bottom + cell * 0.5)) # bottom edge

    # Rank markers on the LEFT and RIGHT edges (same ID on both sides).
    for mid, rank_idx in RANK_MARKER_IDS.items():
        row_from_top = 7 - rank_idx
        ry = int(board_top + (row_from_top + 0.5) * cell)
        stamp(mid, int(board_left - cell * 0.5), ry)   # left edge
        stamp(mid, int(board_right + cell * 0.5), ry)  # right edge

    # Promo squares in all 4 corners. WhitePromo (49) is on black's back-rank
    # side (top of image = rank 8); BlackPromo (50) is on white's side (bottom).
    # Each ID appears twice — once per corner on that player's side — maintaining
    # flip symmetry.
    promo_off = int(cell * 0.6)
    stamp(49, board_left  - promo_off, board_top    - promo_off)  # top-left
    stamp(49, board_right + promo_off, board_top    - promo_off)  # top-right
    stamp(50, board_left  - promo_off, board_bottom + promo_off)  # bottom-left
    stamp(50, board_right + promo_off, board_bottom + promo_off)  # bottom-right

    # A handful of pieces on their starting squares.
    starting_squares: dict[int, tuple[str, int]] = {
        20: ("e", 1),  # WhiteKing1
        16: ("e", 8),  # BlackKing1
        11: ("d", 1),  # WhiteQueen1
        15: ("d", 8),  # BlackQueen1
        1:  ("a", 2),  # WhitePawn1
        3:  ("a", 7),  # BlackPawn1
    }
    for mid, (file_letter, rank_n) in starting_squares.items():
        col = FILES.index(file_letter)
        row_from_top = 8 - rank_n
        px = int(board_left + (col + 0.5) * cell)
        py = int(board_top + (row_from_top + 0.5) * cell)
        stamp(mid, px, py)

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
                piece_positions, promo_positions, warped = result
                log.info("Detected %d piece marker(s): %s",
                         len(piece_positions), piece_positions)
                if promo_positions:
                    log.info("Promo marker(s): %s",
                             {PROMO_NAMES[mid]: pos
                              for mid, pos in promo_positions.items()})

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
