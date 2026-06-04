#!/usr/bin/env python3

import threading
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionClient
from rclpy.duration import Duration

# Standard Stretch hardware messages
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
import re

# TF2 for looking up the ArUco marker
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

# Your custom service
from interfaces.srv import Move
from interfaces.srv import Align
from std_srvs.srv import Trigger


class ChessDriver(Node):
    def __init__(self):
        super().__init__('aruco_base_align_node')

        # 1. Concurrency Setup
        # A ReentrantCallbackGroup allows callbacks to execute in parallel
        self.cb_group = ReentrantCallbackGroup()

        # 2. Configuration
        self.robot_frame = 'base_link'

        # 3. TF2 Setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 4. Joint State Tracking
        self.current_joint_states = {}
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10,
            callback_group=self.cb_group
        )

        # 5. Action Client for Robot Movement
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/stretch_controller/follow_joint_trajectory',
            callback_group=self.cb_group
        )

        # 6. Align Service Server
        self.move_service = self.create_service(
            Move,
            '/chess/move',
            self.move_callback,
            callback_group=self.cb_group
        )

        self.align_service = self.create_service(
            Align,
            '/chess/align',
            self.align_callback,
            callback_group=self.cb_group
        )

        # Add this next to your align service setup
        self.reset_service = self.create_service(
            Trigger,
            '/chess/reset',
            self.reset_callback
        )

        self.previousRank = None

        self.get_logger().info('Chess Driver Initialized. Ready to call /chess/move, /chess/reset')

    # =========================================================
    # Callbacks & Helpers
    # =========================================================

    def reset_callback(self, request, response):
        self.get_logger().info("Executing Reset Sequence...")
        
        # Step 1: Retract, Lift, Orient Wrist
        success = self.execute_trajectory(
            ['joint_lift', 'wrist_extension', 'joint_wrist_yaw', 'joint_wrist_pitch', 'joint_gripper_finger_left'],
            [1.1, 0.0, 3.14, -1.57, 0.17],
            duration_sec=6.0
        )
        if not success:
            response.message = "Failed during arm reset."
            response.success = False
            return response

        # Step 2: Base Rotation (if you have a standard rotation to face the board)
        # ... (paste your base rotation logic here) ...
        # --- STEP 2: Rotate Mobile Base ---
        # self.get_logger().info("Step 2: Rotating base to face marker...")
        # t = self.get_marker_transform(request.pos)
        # if t is None:
        #     response.message = f"Could not find marker '{request.pos}' for rotation."
        #     return response

        # # Calculate the angle to the marker using atan2(y, x)
        # dx = t.transform.translation.x
        # dy = t.transform.translation.y
        # angle_to_marker = math.atan2(dy, dx)

        # # In Stretch, the mobile base tracks relative movements via joint_states
        # current_rot = self.current_joint_states.get('rotate_mobile_base', 0.0)
        # target_rot = current_rot + angle_to_marker

        # success = self.execute_trajectory(
        #     ['rotate_mobile_base'],
        #     [target_rot],
        #     duration_sec=3.0
        # )
        # if not success:
        #     response.message = "Failed during base rotation."
        #     return response


        response.message = "Reset complete."
        response.success = True
        return response

    def joint_states_callback(self, msg):
        """Continuously caches the latest joint positions."""
        for name, position in zip(msg.name, msg.position):
            self.current_joint_states[name] = position

    def execute_trajectory(self, joint_names, positions, duration_sec):
        """
        Helper method to build and send a FollowJointTrajectory goal.
        This blocks the current thread until the action completes, 
        but because we use a MultiThreadedExecutor, it won't freeze the node.
        """
        if not self.trajectory_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Trajectory action server not available!")
            return False

        # Build trajectory point
        point = JointTrajectoryPoint()
        point.positions = positions

        point.time_from_start = Duration(seconds=duration_sec).to_msg()

        # Build goal
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = joint_names
        goal_msg.trajectory.points = [point]

        self.get_logger().info(f"Moving joints: {joint_names} to {positions}")

        # Send the goal asynchronously
        send_goal_future = self.trajectory_client.send_goal_async(goal_msg)
        
        # --- FIX: Block thread safely until the goal is accepted ---
        goal_event = threading.Event()
        send_goal_future.add_done_callback(lambda _: goal_event.set())
        goal_event.wait() # The thread sleeps here, Executor keeps spinning

        goal_handle = send_goal_future.result() 
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected by server.")
            return False

        # --- FIX: Block thread safely until the motion completes ---
        result_future = goal_handle.get_result_async()
        result_event = threading.Event()
        result_future.add_done_callback(lambda _: result_event.set())
        result_event.wait() # The thread sleeps here while robot moves
        
        return True

    def get_marker_transform(self, marker_frame):
        """Helper to get the TF transform with a timeout."""
        try:
            # We look up the transform at the latest available time
            t = self.tf_buffer.lookup_transform(
                self.robot_frame,
                marker_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=2.0)
            )
            return t
        except TransformException as ex:
            self.get_logger().error(f"TF Error: {ex}")
            return None

    # =========================================================
    # Main Service Routine
    # =========================================================

    def move_callback(self, request, response):

        # TODO input validation

        # move to first square
        if self.move_to_square(request.start_file, request.start_rank, response=response) is False:
            return response

        # grab
        if self.grab(response) is False:
            return response        

        # move to first square
        if self.move_to_square(request.end_file, request.end_rank, response=response) is False:
            return response
        
        # drop
        if self.drop(response):
            response.message = f"Successfully moved piece."
        return response

    def open_gripper(self, response):
        self.get_logger().info("Opening gripper...")
        gripper_open_constant = 0.17
        success = self.execute_trajectory(
            ['joint_gripper_finger_left'], 
            [gripper_open_constant], 
            duration_sec=5.0
        )
        if not success:
            response.message = "Gripper failed to open."
            response.success = False 
        return success
    
    def close_gripper(self, response):
        self.get_logger().info("Closing gripper...")
        gripper_close_constant = 0.0
        success = self.execute_trajectory(
            ['joint_gripper_finger_left'], 
            [gripper_close_constant], 
            duration_sec=5.0
        )
        if not success:
            response.message = "Gripper failed to close."
            response.success = False
        return success
    
    def lift_lower(self, response):
        self.get_logger().info("Lowering lift...")
        lift_lower_constant = 0.12
        current_lift = self.current_joint_states.get('joint_lift', 0.95)
        target_lift_down = current_lift - lift_lower_constant
        success = self.execute_trajectory(['joint_lift'], [target_lift_down], duration_sec=5.0)
        if not success:
            response.message = "Lift failed to lower."
            response.success = False
        return success
    
    def lift_raise(self, response):
        self.get_logger().info("Raising lift...")
        lift_top_constant = 1.1
        success = self.execute_trajectory(['joint_lift'], [lift_top_constant], duration_sec=5.0)
        if not success:
            response.message = "Lift failed to raise."
            response.success = False
        return success
    
    def grab(self, response):
        self.get_logger().info("Grabbing...")

        if not self.open_gripper(response):
            return False
        
        time.sleep(1)
        
        if not self.lift_lower(response):
            return False
        
        time.sleep(1)
        
        if not self.close_gripper(response):
            return False
        
        time.sleep(1)

        if not self.lift_raise(response):
            return False
        
        return True
        
    def drop(self, response):
        self.get_logger().info("Dropping...")
        
        if not self.lift_lower(response):
            return False
        
        time.sleep(1)

        if not self.open_gripper(response):
            return False
        
        time.sleep(1)

        if not self.lift_raise(response):
            return False
        
        return True


    def align_callback(self, request, response):
        self.get_logger().info("Align...")

        if self.move_to_square(request.file, request.rank, response) is True:
            response.message = "Aligned to square"
        return response


    
    def move_to_square(self, target_file, target_rank, response):

        file = "File" + target_file.upper()
        rank = "Rank" + target_rank
        
        self.get_logger().info(f"Move to Target Rank: '{target_rank}' | Target File: '{target_file}'")

        self.get_logger().info(f"Pulling arm in and computing heuristic for Rank...")

        setup = False
        if self.previousRank is not None:
            setup = self.execute_trajectory(
                ['wrist_extension', 'translate_mobile_base'],
                [0.0, (int(target_rank) - self.previousRank) * 0.07],
                duration_sec=7.0
            )
        else:
            setup = self.execute_trajectory(
                ['wrist_extension'],
                [0.0],
                duration_sec=7.0
            )
        
        if setup is False:
            response.message = "Reset before move failed."
            response.success = False
            return False

        self.get_logger().info(f"Rank adjustment...")
        # --- BASE ALIGNMENT LOOP (RANK) ---
        base_aligned = False
        target_rank_offset = -0.04
        for attempt in range(5):  # Max 5 iterations to prevent infinite loops
            time.sleep(0.5) # Let the camera and TF buffer catch up after moving
            
            t = self.get_marker_transform(rank)
            
            if t is None:
                self.get_logger().info(f"Target '{rank}' not visible. Hopping to closest rank...")
                visible_rank = self.find_closest_visible_rank(rank)
                
                if not visible_rank:
                    response.message = "No ranks visible at all. Cannot navigate."
                    response.success = False
                    return False
                    
                # Move towards the visible rank just to get closer
                t_temp = self.get_marker_transform(visible_rank)
                dx = t_temp.transform.translation.x
                # Optional: offset slightly so we don't crash into the visible rank
                self.execute_trajectory(
                    ['translate_mobile_base'],
                    [dx],
                    duration_sec=5.0
                )
                continue # Loop again and try to find the real target
                
            # The target IS visible! Fine-tune the approach.
            dx = t.transform.translation.x
            error_x = dx - target_rank_offset
            
            if abs(error_x) < 0.005: # 0.5cm tolerance
                self.get_logger().info("Base successfully aligned to Rank!")
                base_aligned = True
                break
                
            self.get_logger().info(f"Fine-tuning base: moving {error_x:.3f}m")
            self.execute_trajectory(
                    ['translate_mobile_base'],
                    [error_x],
                    duration_sec=5.0
            )
            
        if not base_aligned:
            response.message = "Failed to precisely align base within iterations."
            response.success = False
            return False
        
        # --- ARM EXTENSION LOOP (FILE) ---
        arm_aligned = False
        target_file_offset = 0.16 # The 0.2m offset you requested


        # heuristic first
        index = self.get_file_index(file)

        # TODO fix the adjustments for the File

        self.execute_trajectory(['wrist_extension'], [index * 0.07], duration_sec=5.0)

        
        for attempt in range(4):
            time.sleep(0.5) 
            
            t = self.get_marker_transform(file)
            if t is None:
                response.message = f"Lost sight of {file} during arm extension."
                response.success = False
                return False
                
            # DEPENDING ON YOUR CAMERA FRAME, THIS MIGHT BE .x OR .y OR .z
            # Measure how far the marker is from the arm's extension axis
            dy = t.transform.translation.y
            error_y = -1*target_file_offset - dy

            self.get_logger().info(f'error for the lift {error_y}')
            
            # if abs(error_y) < 0.02: # 0.5cm tolerance
            #     self.get_logger().info("Arm successfully aligned to File!")
            #     arm_aligned = True
            #     break
                
            current_ext = self.current_joint_states.get('wrist_extension', 0.0)
            target_ext = current_ext + error_y

            self.get_logger().info(f'target ext is {target_ext}')
            
            # CRITICAL: Clamp the extension to safe hardware limits (0.0 to 0.5 meters)
            target_ext = max(0.0, min(0.5, target_ext))
            
            self.get_logger().info(f"Fine-tuning arm extension to {target_ext:.3f}m")
            self.execute_trajectory(['wrist_extension'], [target_ext], duration_sec=5.0)

            arm_aligned = True
            break
            
        if not arm_aligned:
            response.message = "Failed to align arm properly."
            response.success = False
            return False
        
        self.previousRank = int(target_rank)
        self.get_logger().info("Move completed...")
        return True

    def get_file_index(self, target_file):
        """
        Converts 'FileA' -> 0, 'FileB' -> 1, etc.
        """
        # 1. Extract just the letter part (e.g., "A" from "FileA")
        # This assumes your strings are always 'File' followed by one letter
        letter = target_file.replace("File", "").upper()
        
        if not letter or not letter.isalpha():
            return None
            
        # 2. Convert character to ASCII/Unicode value and subtract 'A'
        # ord('A') is 65, so ord('A') - 65 = 0, ord('B') - 65 = 1, etc.
        file_index = ord(letter[0]) - ord('A')
        
        return file_index
    
    def find_closest_visible_rank(self, target_rank_str):
        """
        Polls the TF buffer for 'Rank1' through 'Rank8'.
        Returns the name of the visible rank that is numerically closest to the target.
        """
        target_num = int(re.search(r'\d+', target_rank_str).group())
        
        visible_ranks = []
        for i in range(1, 9):
            test_frame = f"Rank{i}"
            # Use a tiny timeout just to check if it exists
            if self.get_marker_transform(test_frame) is not None:
                visible_ranks.append(i)
                
        if not visible_ranks:
            return None
            
        # Find the rank number that is closest to our target number
        closest_num = min(visible_ranks, key=lambda x: abs(x - target_num))
        return f"Rank{closest_num}"


def main(args=None):
    rclpy.init(args=args)
    node = ChessDriver()
    
    # CRITICAL: We must use a MultiThreadedExecutor to allow the service 
    # to block while waiting for the robot trajectory actions to complete.
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()