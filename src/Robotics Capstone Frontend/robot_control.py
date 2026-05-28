"""
Accessible Robotic Chess Player — ROS 2 Service Controller
Interfaces with internal robot ROS 2 node to execute physical moves via service calls.
"""

import rclpy
from rclpy.node import Node

# ---------------------------------------------------------------------------
# Replace 'chess_robot_msgs' with your actual ROS 2 package name.
# Ensure your custom .srv types match the request/response structure needed.
# ---------------------------------------------------------------------------
from chess_robot_msgs.srv import ChessMove, ChessPromote


class ChessRobotROSController(Node):
    """
    High-level ROS 2 client for the robot arm.
    Calls specific movement services hosted by the robot's internal node.
    """

    def __init__(self) -> None:
        super().__init__('chess_robot_client')
        
        self.get_logger().info("Initializing Robot Chess Controller...")

        # Initialize Service Clients
        self.cli_move    = self.create_client(ChessMove, 'move')
        self.cli_take    = self.create_client(ChessMove, 'take')
        self.cli_castle  = self.create_client(ChessMove, 'castle')
        self.cli_enpass  = self.create_client(ChessMove, 'EnPass')
        self.cli_promote = self.create_client(ChessPromote, 'promote')

        # Block until all hardware services are online
        self._wait_for_services()
        self.get_logger().info("Hardware connection established. Ready for commands.")

    def _wait_for_services(self) -> None:
        """Helper to ensure all required services are available before proceeding."""
        clients = [
            self.cli_move, 
            self.cli_take, 
            self.cli_castle, 
            self.cli_enpass, 
            self.cli_promote
        ]
        
        for cli in clients:
            while not cli.wait_for_service(timeout_sec=2.0):
                self.get_logger().warn(f"Waiting for robot service: '{cli.srv_name}'...")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def move(self, notation: str):
        """Standard move without capture. (e.g., 'e2e4')"""
        self.get_logger().info(f"Command: MOVE -> {notation}")
        req = ChessMove.Request()
        req.notation = notation
        return self._call_service_sync(self.cli_move, req)

    def take(self, notation: str):
        """Move that captures an opponent piece. (e.g., 'e4d5')"""
        self.get_logger().info(f"Command: TAKE -> {notation}")
        req = ChessMove.Request()
        req.notation = notation
        return self._call_service_sync(self.cli_take, req)

    def castle(self, notation: str):
        """Castling move. (e.g., 'e1g1' or 'O-O')"""
        self.get_logger().info(f"Command: CASTLE -> {notation}")
        req = ChessMove.Request()
        req.notation = notation
        return self._call_service_sync(self.cli_castle, req)

    def en_passant(self, notation: str):
        """En Passant capture. (e.g., 'e5d6')"""
        self.get_logger().info(f"Command: EN PASSANT -> {notation}")
        req = ChessMove.Request()
        req.notation = notation
        return self._call_service_sync(self.cli_enpass, req)

    def promote(self, square: str, piece: str):
        """Pawn promotion. Piece must be 'Q' or 'K'."""
        self.get_logger().info(f"Command: PROMOTE -> {square} to {piece.upper()}")
        req = ChessPromote.Request()
        req.notation = square
        req.piece = piece.upper()
        return self._call_service_sync(self.cli_promote, req)

    # ------------------------------------------------------------------ #
    # Internal Execution                                                   #
    # ------------------------------------------------------------------ #

    def _call_service_sync(self, client, request):
        """
        Synchronous wrapper for service calls. 
        Spins the node until the robot finishes the physical move and responds.
        """
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        result = future.result()
        if result is not None:
            self.get_logger().info(f"Hardware execution successful.")
            return result
        else:
            self.get_logger().error(f"Hardware execution failed.")
            return None


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    
    controller = ChessRobotROSController()

    try:
        # These will block until the physical robot completes the move
        controller.move("e2e4")
        controller.take("e4d5")
        controller.castle("e1g1")
        controller.en_passant("e5d6")
        controller.promote("e8", "Q")

    except KeyboardInterrupt:
        controller.get_logger().info("Interrupted by user, shutting down.")
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()