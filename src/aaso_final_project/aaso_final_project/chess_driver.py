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

import subprocess

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
from interfaces.srv import Promote
from interfaces.srv import Castle
from interfaces.srv import EnPassant
from std_srvs.srv import Trigger


class ChessDriver(Node):

    # TODO add promotion
    # TODO Make lift grab height use aruco markers?
    # TODO Look for speed ups?
    # TODO Make a system for the robot backing up from the table if needed
    # TODO If the robot cannot see any ranks for rank adjustment then backup?
    # TODO For file/rank adjustment use other rank / file markers with an offset?
    # TODO detect and auto align to the table if needed during moving
    # TODO switch the force reset wrist to use aruco markers instead of hitting 0.0 + error?
    # TODO Robot slowly approaches the table over time and also starts to rotate slightly due to droop of the arm


    # Constants -- to be tuned depending
    GRIPPER_OPEN = 0.18
    GRIPPER_CLOSED = -0.05

    LIFT_TOP = 1.1
    LIFT_PICKUP_HEIGHT = 0.97
    LIFT_DROP_HEIGHT = 0.965
    LIFT_ADJUST_HEIGHT = 1.03

    WRIST_YAW_DISCARD = 0.0
    WRIST_YAW_NORMAL = math.pi

    WRIST_PITCH_DISCARD = 0.0
    WRIST_PITCH_NORMAL = -math.pi / 2

    WRIST_EXTENSION_DISCARD = 0.42
    WRIST_EXTENSION_RESET = 0.0

    WRIST_RETRACTION_ERROR = 0.04

    ONE_RANK_DISTANCE = 0.071

    RANK_ADJUSTMENT_OFFSET = -0.005
    FILE_ADJUSTMENT_OFFSET = 0.034

    RANK_ERROR_ALLOWED = 0.005
    FILE_ERROR_ALLOWED = 0.005

    FINGER_ADJUSTMENT_OFFSET = 0.0

    MOVE_AWAY_FROM_TABLE_DISTANCE = -0.2
    MOVE_AWAY_FROM_TABLE_ANGLE = -math.pi / 5.0

    SAFE_TABLE_CLEARANCE = 0.22
    TABLE_DISTANCE_ERROR_ALLOWED = 0.02
    HEADING_ERROR_ALLOWED = 0.03
    HEADING_OFFSET = 0.05


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

        self.move_service = self.create_service(
            Move,
            '/chess/move',
            self.move_callback,
            callback_group=self.cb_group
        )

        self.take_service = self.create_service(
            Move,
            '/chess/take',
            self.take_callback,
            callback_group=self.cb_group
        )

        self.align_service = self.create_service(
            Align,
            '/chess/align',
            self.align_callback,
            callback_group=self.cb_group
        )

        self.reset_service = self.create_service(
            Trigger,
            '/chess/reset',
            self.reset_callback
        )

        self.promote_service = self.create_service(
            Promote,
            '/chess/promote',
            self.promote_callback
        )

        self.castle_service = self.create_service(
            Castle,
            '/chess/castle',
            self.castle_callback
        )

        self.enpassant_service = self.create_service(
            EnPassant,
            '/chess/enpassant',
            self.enpassant_callback
        )

        self.previousRank = None

        self.get_logger().info('Chess Driver Initialized. Ready to call /chess/move, /chess/reset, /chess/take, /chess/align, /chess/promote, /chess/enpassant, or /chess/castle')
        self.get_logger().info(f'The ROS Bridge (if running) Host IP should be: {self.get_ip_address()} with port 9090')


    # =========================================================
    # Main Service Callbacks
    # =========================================================

    def reset_callback(self, request, response):
        self.get_logger().info("Executing Reset Sequence...")

        if not self.gripper_reset_position(response):
            return response

        if not self.align_to_table(response):
            return response

        if self.align_heading_to_table(response):
            response.message = "Reset and alignment complete."
            response.success = True
        return response
    
    def move_callback(self, request, response):

        self.get_logger().info('Request received for a move...')

        # TODO Input validation?
        
        if self.move(response, request.start_file, request.start_rank, request.end_file, request.end_rank, request.start_piece):
            response.message = f"Successfully moved piece."
            response.success = True
        return response
    
    def promote_callback(self, request, response):

        # TODO DOES THIS WORK? I DON'T KNOW

        self.get_logger().info('Promote received for a move...')

        if self.remove_piece(response, request.start_file, request.start_rank, request.start_piece) is False:
            return response

        if request.end_piece != 'empty':
            if self.remove_piece(response, request.end_file, request.end_rank, request.end_piece) is False:
                return response

        if self.move(response, request.promo_file, request.promo_rank, request.end_file, request.end_rank, request.promote_piece) is False:
            return response

        response.message = f"Successfully promoted piece."
        response.success = True
        return response
    
    def castle_callback(self, request, response):

        self.get_logger().info('Castle received for a move...')

        if self.move(response, request.start_file_k, request.start_rank_k, request.end_file_k, request.end_rank_k, request.king_piece) is False:
            return response
        
        if self.move(response, request.start_file_r, request.start_rank_r, request.end_file_r, request.end_rank_r, request.rook_piece):
            response.message = f"Successfully castled piece."
            response.success = True
            
        return response
    
    def enpassant_callback(self, request, response):

        self.get_logger().info('EnPassant received for a move...')

        if self.remove_piece(response, request.file_l, request.rank_l, request.lose_pawn) is False:
            return response
        
        if self.move(response, request.start_file_w, request.start_rank_w, request.end_file_w, request.end_rank_w, request.win_pawn):
            response.message = f"Successfully EnPassant piece."
            response.success = True
            
        return response
    
    def align_callback(self, request, response):
        self.get_logger().info("Align...")

        if self.move_to_square(request.file, request.rank, response) is True:
            response.message = "Aligned to square"
        return response
    
    def take_callback(self, request, response):

        self.get_logger().info('Request received for a take...')

        if self.remove_piece(response, request.end_file, request.end_rank, request.end_piece) is False:
            return response

        if self.move(response, request.start_file, request.start_rank, request.end_file, request.end_rank, request.start_piece):
            response.message = f"Successfully moved piece."
            response.success = True
        return response
    
    # =========================================================
    # Callbacks & Helpers
    # =========================================================

    def get_ip_address(self):
        try:
            # Run the command and capture the output
            # 'capture_output=True' stores stdout and stderr
            # 'text=True' ensures the output is returned as a string rather than bytes
            result = subprocess.run(['hostname', '-I'], capture_output=True, text=True, check=True)
            
            # The result includes a trailing newline, so we use .strip()
            return result.stdout.strip()
            
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}")
        except FileNotFoundError:
            print("The 'hostname' command was not found.")
        
        return None

    def joint_states_callback(self, msg):
        """Continuously caches the latest joint positions."""
        for name, position in zip(msg.name, msg.position):
            self.current_joint_states[name] = position

    def get_yaw_from_quaternion(self, q):
        """
        Converts a geometry_msgs Quaternion to yaw (rotation around Z-axis) in radians.
        """
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def align_heading_to_table(self, response):
        """
        Reads the orientation of a visible rank marker and rotates the base to perfectly square up.
        """
        self.get_logger().info("Squaring up base heading with the table...")
        
        # We give it a couple of attempts to fine-tune, similar to your rank alignment
        for attempt in range(3):
            time.sleep(1.0) # Let camera and TF settle
            
            marker_t = self.get_visible_rank_transform()
            if marker_t is None:
                return self.fail(response, "Lost sight of markers. Cannot align heading.")
                
            # Extract the yaw error from the marker's orientation
            q = marker_t.transform.rotation
            yaw_error = self.get_yaw_from_quaternion(q) + math.pi / 2 + self.HEADING_OFFSET
            
            # Note: Depending on exactly how your ArUco markers are defined in the TF tree, 
            # you MIGHT need to add or subtract math.pi/2 to yaw_error if the robot 
            # tries to align sideways instead of facing the table.
            
            self.get_logger().info(f"Heading error is {yaw_error:.3f} rad.")
            
            if abs(yaw_error) < self.HEADING_ERROR_ALLOWED:
                self.get_logger().info("Base is successfully flush with the table!")
                return True
                
            # Rotate the base to correct the error
            current_rot = self.current_joint_states.get('rotate_mobile_base', 0.0)
            target_rot = current_rot + yaw_error
            
            self.get_logger().info(f"Correcting heading by rotating base to {target_rot:.3f} rad")
            self.execute_trajectory(
                ['rotate_mobile_base'],
                [target_rot],
                duration_sec=3.0
            )
            
        return self.fail(response, "Failed to align heading to table.")
    

    def fail(self, response, msg):
        self.get_logger().error(msg)
        response.message = msg
        response.success = False
        return False

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

    def get_marker_transform(self, marker_frame, robot_frame):
        """Helper to get the TF transform with a timeout."""
        try:
            # We look up the transform at the latest available time
            t = self.tf_buffer.lookup_transform(
                robot_frame,
                marker_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=2.0)
            )
            return t
        except TransformException as ex:
            self.get_logger().error(f"TF Error: {ex}")
            return None
    
    def get_visible_rank_transform(self):
        """
        Returns the transform of the first visible rank marker, or None if none are seen.
        """
        # Using your specific non-sequential index order
        for i in (5, 4, 6, 3, 7, 2, 1):
            test_frame = f"Rank{i}W"
            t = self.get_marker_transform(test_frame, "base_link")
            if t is not None:
                self.get_logger().info(f'{test_frame} was found.')
                return t
        return None

    def align_to_table(self, response):
        # reset wrist
        self.force_reset_wrist()

        # Step 2: Handle Table Clearance
        marker_t = self.get_visible_rank_transform()

        odd = 1
        # if we know which direction will hit the table, do the opposite
        if self.previousRank is not None:
            if self.previousRank < 5:
                odd = -1
        
        # If we are totally blind, back up in standard increments until we see the table
        while marker_t is None:
            self.get_logger().info('No ranks visible, blindly backing away from table...')
            if self.backup_from_table(odd) is False:
                response.message = "Failed during blind back away."
                response.success = False
                return False
            odd *= -1
            marker_t = self.get_visible_rank_transform()

        # Now that we see a marker, calculate exactly how much further we need to go
        # The perpendicular distance to the table is roughly the Y-axis value of the marker
        current_y_dist = abs(marker_t.transform.translation.y)
        delta_dist = self.SAFE_TABLE_CLEARANCE - current_y_dist
        
        if delta_dist > self.TABLE_DISTANCE_ERROR_ALLOWED: # Only trigger if we are more than 2cm too close
            self.get_logger().info(f"Currently {current_y_dist:.3f}m from table. Need {self.SAFE_TABLE_CLEARANCE}m.")
            
            # Apply trigonometry: hypotenuse = opposite / sin(theta)
            calc_trans = delta_dist / math.sin(abs(self.MOVE_AWAY_FROM_TABLE_ANGLE))
            
            # Match the sign of your standard distance constant to ensure we move backward
            calc_trans = math.copysign(calc_trans, self.MOVE_AWAY_FROM_TABLE_DISTANCE)
            
            self.get_logger().info(f"Calculated hypotenuse translation: {calc_trans:.3f}m")
            
            if self.backup_from_table(odd=1, calculated_distance=calc_trans) is False:
                response.message = "Failed during precision back away."
                response.success = False
                return False
        else:
            self.get_logger().info("Robot is already at a safe distance from the table.")
        
        return True
    
    def move(self, response, start_file, start_rank, end_file, end_rank, piece):
        if self.move_to_square(start_file, start_rank, response=response) is False:
            return False

        if self.grab(response, piece) is False:
            return False     

        if self.move_to_square(end_file, end_rank, response=response) is False:
            return False
        
        return self.drop(response)
    
    def remove_piece(self, response, file, rank, piece):
        if self.move_to_square(file, rank, response=response) is False:
            return False

        if self.grab(response, piece) is False:
            return False
        
        if self.gripper_discard_position(response=response) is False:
            return False
        
        if self.open_gripper(response) is False:
            return False
        
        return self.gripper_reset_position(response)
    
    def backup_from_table(self, odd, calculated_distance=None):
        """
        Rotates the mobile base, translates, and rotates back.
        If calculated_distance is provided, it uses that instead of the standard distance.
        """
        self.get_logger().info("Executing backup maneuver...")

        # --- Step 1: Turn ---
        current_rot = self.current_joint_states.get('rotate_mobile_base', 0.0)
        target_rot = current_rot + odd * self.MOVE_AWAY_FROM_TABLE_ANGLE 
        
        self.get_logger().info(f"Turning base to {target_rot:.3f} rad")
        success1 = self.execute_trajectory(['rotate_mobile_base'], [target_rot], duration_sec=4.0)
        
        if not success1:
            return False

        time.sleep(0.5)

        # --- Step 2: Translate ---
        current_trans = self.current_joint_states.get('translate_mobile_base', 0.0)
        
        # USE CALCULATED DISTANCE IF PROVIDED
        if calculated_distance is not None:
            target_trans = current_trans + odd * calculated_distance
        else:
            target_trans = current_trans + odd * self.MOVE_AWAY_FROM_TABLE_DISTANCE
        
        self.get_logger().info(f"Translating base to {target_trans:.3f} m")
        success2 = self.execute_trajectory(['translate_mobile_base'], [target_trans], duration_sec=3.0)
        
        if not success2:
            return False

        time.sleep(0.5)

        # --- Step 3: Turn back ---
        current_rot2 = self.current_joint_states.get('rotate_mobile_base', 0.0)
        target_rot_back = current_rot2 - odd * self.MOVE_AWAY_FROM_TABLE_ANGLE
        
        self.get_logger().info(f"Returning base rotation to {target_rot_back:.3f} rad")
        success3 = self.execute_trajectory(['rotate_mobile_base'], [target_rot_back], duration_sec=4.0)
        
        if not success3:
            return False

        self.get_logger().info("Successfully completed backup maneuver.")
        return True

    def open_gripper(self, response):
        self.get_logger().info("Opening gripper...")
        success = self.execute_trajectory(
            ['joint_gripper_finger_left'], 
            [self.GRIPPER_OPEN], 
            duration_sec=5.0
        )
        if not success:
            response.message = "Gripper failed to open."
            response.success = False 
        return success
    
    def gripper_discard_position(self, response):
        self.get_logger().info("Discard positioning gripper...")
        success1 = self.execute_trajectory(
            ['wrist_extension', 'joint_wrist_yaw', 'joint_wrist_pitch'],
            [self.WRIST_EXTENSION_DISCARD, self.WRIST_YAW_DISCARD, self.WRIST_PITCH_DISCARD],
            duration_sec=6.0
        )

        success2 = self.execute_trajectory(
            ['translate_mobile_base'],
            [0.0],
            duration_sec=6.0
        )

        if not success1 or not success2:
            response.message = "Robot failed to go into discard position."
            response.success = False
        return success1
    
    def gripper_reset_position(self, response):
        self.get_logger().info("Reset positioning gripper...")
        
        s1 = self.execute_trajectory(
            ['joint_lift', 'joint_wrist_yaw', 'joint_wrist_pitch', 'joint_gripper_finger_left'],
            [self.LIFT_TOP, self.WRIST_YAW_NORMAL, self.WRIST_PITCH_NORMAL, self.GRIPPER_OPEN],
            duration_sec=6.0
        )

        s2 = self.force_reset_wrist()

        if not s1 or not s2:
            self.fail(response, "Gripper failed to go into reset position.")
        
        return s1 and s2
    
    def close_gripper(self, response):
        self.get_logger().info("Closing gripper...")
        success = self.execute_trajectory(
            ['joint_gripper_finger_left'], 
            [self.GRIPPER_CLOSED], 
            duration_sec=5.0
        )
        if not success:
            response.message = "Gripper failed to close."
            response.success = False
        return success
    
    def lift_lower(self, response, height):
        self.get_logger().info("Lowering lift...")
        success = self.execute_trajectory(['joint_lift'], [height], duration_sec=5.0)
        if not success:
            response.message = "Lift failed to lower."
            response.success = False
        return success
    
    def lift_raise(self, response):
        self.get_logger().info("Raising lift...")
        success = self.execute_trajectory(['joint_lift'], [self.LIFT_TOP], duration_sec=5.0)
        if not success:
            response.message = "Lift failed to raise."
            response.success = False
        return success
    

    def force_reset_wrist(self):
        success = False
        self.get_logger().info('Force reset wrist...')
        for atempts in range(15):
            cur = self.get_wrist_position()

            self.get_logger().info(f'Current wrist position: {cur}')

            if cur < self.WRIST_RETRACTION_ERROR:
                success = True
                break

            self.execute_trajectory(
            ['wrist_extension'],
            [self.WRIST_EXTENSION_RESET],
            duration_sec=6.0
            )

            time.sleep(1)

        return success

    
    def grab(self, response, piece):
        self.get_logger().info("Grabbing...")
        self.get_logger().info(f"Aligning to {piece}...")

        if not self.open_gripper(response):
            return False
        
        time.sleep(1)
        
        if not self.lift_lower(response, self.LIFT_ADJUST_HEIGHT):
            return False
        
        # 1. Get transforms of all three markers relative to the robot's base
        # lookup_transform syntax: (target_frame, source_frame, time)
        # Using base_link as target_frame gives us the marker coordinates IN the base_link frame.
        trans_left = self.get_marker_transform('link_aruco_fingertip_left', 'base_link')
        trans_right = self.get_marker_transform('link_aruco_fingertip_right', 'base_link')
        trans_target = self.get_marker_transform(piece, 'base_link')

        # 2. Calculate the midpoint of the two fingertip markers (we only strictly need X for base translation)
        mid_x = (trans_left.transform.translation.x + trans_right.transform.translation.x) / 2.0
        
        # 3. Calculate the relative position of the target with respect to the fingertip midpoint along the X-axis
        # This gives us the forward/backward error in meters
        dx = trans_target.transform.translation.x - mid_x
        
        self.get_logger().info(f"Target is off by {dx:.3f} meters along X. Translating base...")

        # 4. Command the base translation to close the X-axis gap
        # translate_mobile_base acts as a virtual prismatic joint tracking odometry
        current_trans = self.current_joint_states.get('translate_mobile_base', 0.0)
        target_trans = current_trans + dx

        success = self.execute_trajectory(
            ['translate_mobile_base'],
            [float(target_trans)],
            duration_sec=3.0
        )

        time.sleep(1)

        # 2. Calculate the midpoint of the two fingertip markers along the Y-axis
        mid_y = (trans_left.transform.translation.y + trans_right.transform.translation.y) / 2.0

        # 3. Calculate the relative position of the target with respect to the fingertip midpoint
        # This gives us the in/out error in meters
        dy = trans_target.transform.translation.y - mid_y
        
        self.get_logger().info(f"Target is off by {dy:.3f} meters along Y. Adjusting arm extension...")

        # 4. Calculate the current total arm extension
        current_extension = self.get_wrist_position()

        # 5. Calculate the new target extension
        # NOTE: Depending on how the base_link axes are oriented relative to the arm's extension direction 
        # in your specific TF tree, you might need to subtract dy instead of adding it.
        target_extension = current_extension - dy

        # Apply safety bounds (The Stretch arm has a physical max extension of roughly 0.52 meters)
        target_extension = max(0.0, min(0.52, target_extension))

        # 7. Execute the trajectory to move the individual arm joints
        success = self.execute_trajectory(['wrist_extension'], [target_extension], duration_sec=5.0)

        # TODO ADD A SAFTEY CHECK? FOR OTHER PIECES
        
        time.sleep(1)

        if not self.lift_lower(response, self.LIFT_PICKUP_HEIGHT):
            return False
        
        time.sleep(2)
        
        if not self.close_gripper(response):
            return False
        
        time.sleep(1)

        if not self.lift_raise(response):
            return False
        
        return True
    
    def get_wrist_position(self):
        return self.current_joint_states.get('joint_arm_l0', 0.0) + self.current_joint_states.get('joint_arm_l1', 0.0) + self.current_joint_states.get('joint_arm_l2', 0.0) + self.current_joint_states.get('joint_arm_l3', 0.0)
        
    def drop(self, response):
        self.get_logger().info("Dropping...")
        
        if not self.lift_lower(response, self.LIFT_DROP_HEIGHT):
            return False
        
        time.sleep(1)

        if not self.open_gripper(response):
            return False
        
        time.sleep(1)

        if not self.lift_raise(response):
            return False
        
        return True
    
    def move_to_square(self, target_file, target_rank, response):

        file = "File" + target_file.upper()

        off_by_one_rank_offset = 0.0
        rank = ""
        if target_rank == 'WhitePromo' or target_rank == 'BlackPromo':
            off_by_one_rank_offset = self.ONE_RANK_DISTANCE / 2.0
        else:
            if int(target_rank) >= 5:
                file += "B"
            else:
                file += "W"

            rank = "Rank" + target_rank + "W"

            if rank == "Rank8W":
                rank = "Rank7W"
                off_by_one_rank_offset = self.ONE_RANK_DISTANCE
        
        self.get_logger().info(f"Move to Target Rank: '{target_rank}' | Target File: '{target_file}'")

        self.get_logger().info(f"Pulling arm in and computing heuristic for Rank...")

        if self.previousRank is not None:
            cur = self.current_joint_states.get('translate_mobile_base', 0.0)
            setup = self.execute_trajectory(
                ['translate_mobile_base'],
                [(int(target_rank) - self.previousRank) * self.ONE_RANK_DISTANCE + cur],
                duration_sec=7.0
            )
        
        if self.force_reset_wrist() is False:
            self.get_logger().info('Force write movement failed!')
            response.message = "Reset before move failed."
            response.success = False
            return False

        self.get_logger().info(f"Rank adjustment...")
        # --- BASE ALIGNMENT LOOP (RANK) ---
        base_aligned = False
        for attempt in range(4):  # Max 5 iterations to prevent infinite loops
            time.sleep(1.0) # Let the camera and TF buffer catch up after moving
            
            t = self.get_marker_transform(rank, "base_link")
            
            if t is None:
                self.get_logger().info(f"Target '{rank}' not visible. Hopping to closest rank...")

                # Move towards the visible rank just to get closer
                t_temp = self.get_marker_transform("Rank5W", "base_link")
                dx = t_temp.transform.translation.x
                cur = self.current_joint_states.get('translate_mobile_base', 0.0)
                # Optional: offset slightly so we don't crash into the visible rank
                self.execute_trajectory(
                    ['translate_mobile_base'],
                    [dx + cur],
                    duration_sec=5.0
                )
                time.sleep(0.5)
                continue # Loop again and try to find the real target
                
            # The target IS visible! Fine-tune the approach.
            error_x = t.transform.translation.x - self.RANK_ADJUSTMENT_OFFSET + off_by_one_rank_offset
            
            if abs(error_x) < self.RANK_ERROR_ALLOWED: # 0.5cm tolerance
                self.get_logger().info("Base successfully aligned to Rank!")
                base_aligned = True
                break
                
            self.get_logger().info(f"Fine-tuning base: moving {error_x:.3f}m")
            cur = self.current_joint_states.get('translate_mobile_base', 0.0)
            self.execute_trajectory(
                    ['translate_mobile_base'],
                    [error_x + cur],
                    duration_sec=5.0
            )
            
        if not base_aligned:
            self.get_logger().info(f'FAILED TO ALIGN BASE FOR RANK')
            response.message = "Failed to precisely align base within iterations."
            response.success = False
            return False
        
        self.previousRank = int(target_rank)
        
        # --- ARM EXTENSION LOOP (FILE) ---
        arm_aligned = False
        # heuristic first
        index = self.get_file_index(file)

        # TODO fix the adjustments for the File

        self.execute_trajectory(['wrist_extension'], [(index + 0.1) * self.ONE_RANK_DISTANCE], duration_sec=5.0)

        time.sleep(2)
        
        for attempt in range(5):
            time.sleep(1)
            
            t = self.get_marker_transform(file, "gripper_camera_color_optical_frame")
            if t is None:
                response.message = f"Lost sight of {file} during arm extension."
                response.success = False
                return False
                
            # DEPENDING ON YOUR CAMERA FRAME, THIS MIGHT BE .x OR .y OR .z
            # Measure how far the marker is from the arm's extension axis
            error_y = t.transform.translation.y + self.FILE_ADJUSTMENT_OFFSET

            self.get_logger().info(f'error for the lift {error_y}')
            
            if abs(error_y) < self.FILE_ERROR_ALLOWED: # 2cm tolerance
                self.get_logger().info("Arm successfully aligned to File!")
                arm_aligned = True
                break

            current_ext = self.get_wrist_position()                
            self.get_logger().info(f'current wrist extension is {current_ext}')

            target_ext =  error_y + current_ext

            self.get_logger().info(f'target ext is {target_ext}')
            
            self.get_logger().info(f"Fine-tuning arm extension to {target_ext:.3f}m")
            self.execute_trajectory(['wrist_extension'], [target_ext], duration_sec=5.0)
            
        if not arm_aligned:
            self.get_logger().info(f'FAILED TO ALIGN ARM')
            response.message = "Failed to align arm properly."
            response.success = False
            return False
        
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
    
    def is_a_rank_visible(self):
        for i in (5, 4, 6, 3):
            test_frame = f"Rank{i}W"
            self.get_logger().info(f'Testing if {test_frame} is visible...')
            if self.get_marker_transform(test_frame, "base_link") is not None:
                self.get_logger().info(f'{test_frame} was found.')
                return True
        return False


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