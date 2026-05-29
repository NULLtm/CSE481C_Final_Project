"""
ArUco Detection Node — aaso_final_project

Subscribes to the Stretch head camera, runs the ArUco perception pipeline
(imported from the frontend's perception.py), and publishes the detected
piece→square mapping as a JSON string on /chess/board_state.

Also provides the /AlgNotation service so any ROS 2 node or rosbridge client
can request a human-readable description of the current board state.

Run on the Stretch:
    ros2 run aaso_final_project aruco_node
"""

import json
import os
import sys

import rclpy
from std_msgs.msg import String

import hello_helpers.hello_misc as hm

# ---------------------------------------------------------------------------
# Import perception pipeline from the frontend package
# ---------------------------------------------------------------------------
# Both this file and perception.py live in the same git repo.  On the Stretch
# the full repo is cloned, so we can locate perception.py via a relative path.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
_FRONTEND_DIR = os.path.join(_REPO_ROOT, "src", "Robotics Capstone Frontend")
if os.path.isdir(_FRONTEND_DIR) and _FRONTEND_DIR not in sys.path:
    sys.path.insert(0, _FRONTEND_DIR)

try:
    import cv2
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image
    from perception import (
        detect_board_state,
        _build_detector,
        PIECE_NAMES,
        PROMO_NAMES,
    )
    _PERCEPTION_OK = True
except ImportError as _e:
    _PERCEPTION_OK = False
    _IMPORT_ERR = str(_e)

# ---------------------------------------------------------------------------
# Import generated service (built from srv/AlgNotation.srv)
# ---------------------------------------------------------------------------
try:
    from aaso_final_project.srv import AlgNotation
    _SRV_OK = True
except ImportError:
    _SRV_OK = False

CAMERA_TOPIC      = "/camera/color/image_raw"
BOARD_STATE_TOPIC = "/chess/board_state"   # std_msgs/String  (JSON payload)


class ArucoDetectionNode(hm.HelloNode):
    """
    Detects ArUco markers from the Stretch head camera and publishes
    the current piece→square mapping to /chess/board_state.

    Published JSON schema:
        {
            "pieces": {"<marker_id>": "<square>", ...},   // e.g. {"20": "e1"}
            "promos": {"<marker_id>": [img_x, img_y], ...}
        }
    """

    def __init__(self) -> None:
        hm.HelloNode.__init__(self)
        self._cv_bridge          = None
        self._detector           = None
        self._board_state_pub    = None
        self._latest_pieces: dict[int, str] = {}

    def main(self) -> None:
        hm.HelloNode.main(
            self,
            "aruco_chess_node",
            "aruco_chess_node",
            wait_for_first_pointcloud=False,
        )

        if not _PERCEPTION_OK:
            self.get_logger().error(
                "Cannot import perception pipeline: %s\n"
                "Ensure opencv-contrib-python is installed and perception.py "
                "is reachable at: %s",
                _IMPORT_ERR, _FRONTEND_DIR,
            )
            return

        self._cv_bridge = CvBridge()
        self._detector  = _build_detector()

        self._board_state_pub = self.create_publisher(String, BOARD_STATE_TOPIC, 10)
        self.create_subscription(Image, CAMERA_TOPIC, self._image_cb, 10)

        if _SRV_OK:
            self.create_service(AlgNotation, "AlgNotation", self._alg_notation_cb)
            self.get_logger().info("AlgNotation service advertised.")
        else:
            self.get_logger().warning(
                "AlgNotation service not available — rebuild the package so "
                "srv/AlgNotation.srv is generated."
            )

        self.get_logger().info(
            "ArucoDetectionNode ready.\n"
            "  Subscribing:  %s\n"
            "  Publishing:   %s",
            CAMERA_TOPIC, BOARD_STATE_TOPIC,
        )

        try:
            rclpy.spin(self)
        except KeyboardInterrupt:
            pass
        finally:
            self.destroy_node()
            rclpy.try_shutdown()

    # ------------------------------------------------------------------
    # Camera callback
    # ------------------------------------------------------------------

    def _image_cb(self, msg: "Image") -> None:
        """Convert a ROS Image to OpenCV, run detection, publish board state."""
        try:
            frame = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning("cv_bridge conversion failed: %s", exc)
            return

        result = detect_board_state(frame, self._detector)
        if result is None:
            return   # not enough fiducials visible yet

        pieces, promos, _warped = result
        self._latest_pieces = pieces

        payload = {
            "pieces": {str(pid): sq for pid, sq in pieces.items()},
            "promos": {str(mid): [round(x, 1), round(y, 1)]
                       for mid, (x, y) in promos.items()},
        }
        self._board_state_pub.publish(String(data=json.dumps(payload)))

    # ------------------------------------------------------------------
    # AlgNotation service
    # ------------------------------------------------------------------

    def _alg_notation_cb(self, request, response):
        """
        Return the detected board state as human-readable text.
        The `pos` request field is accepted but currently unused; extend
        this callback to parse SAN or UCI queries as needed.
        """
        if not self._latest_pieces:
            response.pos = "No pieces detected."
            return response

        lines = sorted(
            f"{PIECE_NAMES.get(pid, f'ID{pid}')}: {sq}"
            for pid, sq in self._latest_pieces.items()
        )
        response.pos = "\n".join(lines)
        return response


def main() -> None:
    node = ArucoDetectionNode()
    node.main()


if __name__ == "__main__":
    main()
