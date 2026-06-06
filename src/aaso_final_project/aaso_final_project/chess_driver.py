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
from std_srvs.srv import Trigger


class ChessDriver(Node):

    # Constants -- to be tuned depending

    GRIPPER_OPEN = 0.11
    GRIPPER_CLOSED = -0.05

    LIFT_TOP = 1.1
    LIFT_PICKUP_HEIGHT = 0.97
    LIFT_DROP_HEIGHT = 0.96
    LIFT_ADJUST_HEIGHT = 1.03

    WRIST_YAW_DISCARD = 0.0
    WRIST_YAW_NORMAL = math.pi

    WRIST_PITCH_DISCARD = 0.0
    WRIST_PITCH_NORMAL = -math.pi / 2

    WRIST_EXTENSION_DISCARD = 0.4
    WRIST_EXTENSION_RESET = 0.0

    WRIST_RETRACTION_ERROR = 0.04

    ONE_RANK_DISTANCE = 0.071

    RANK_ADJUSTMENT_OFFSET = 0.001
    FILE_ADJUSTMENT_OFFSET = 0.034

    RANK_ERROR_ALLOWED = 0.005
    FILE_ERROR_ALLOWED = 0.01

    FINGER_ADJUSTMENT_OFFSET = 0.0


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

        self.previousRank = None

        self.get_logger().info('Chess Driver Initialized. Ready to call /chess/move, /chess/reset, /chess/take, /chess/align')
        self.get_logger().info(f'The Host IP should be: {self.get_ip_address()} with port 9090')

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

    # =========================================================
    # Main Service Routine
    # =========================================================

    def reset_callback(self, request, response):
        self.get_logger().info("Executing Reset Sequence...")
        
        # Step 1: Retract, Lift, Orient Wrist
        success = self.execute_trajectory(
            ['joint_lift', 'wrist_extension', 'joint_wrist_yaw', 'joint_wrist_pitch', 'joint_gripper_finger_left'],
            [self.LIFT_TOP, self.WRIST_EXTENSION_RESET, self.WRIST_YAW_NORMAL, self.WRIST_PITCH_NORMAL, self.GRIPPER_OPEN],
            duration_sec=6.0
        )

        self.force_reset_wrist()

        if not success:
            response.message = "Failed during arm reset."
            response.success = False
            return response

        response.message = "Reset complete."
        response.success = True
        return response

    def move_callback(self, request, response):

        self.get_logger().info('Request received for a move...')

        # TODO Input validation?
        
        if self.move(response, request.start_file, request.start_rank, request.end_file, request.end_rank, request.start_piece):
            response.message = f"Successfully moved piece."
            response.success = True
        return response
    
    def move(self, response, start_file, start_rank, end_file, end_rank, piece):
        if self.move_to_square(start_file, start_rank, response=response) is False:
            return False

        if self.grab(response, piece) is False:
            return False     

        if self.move_to_square(end_file, end_rank, response=response) is False:
            return False
        
        return self.drop(response)
    
    def take_callback(self, request, response):

        self.get_logger().info('Request received for a take...')

        if self.move_to_square(request.end_file, request.end_rank, response=response) is False:
            return response

        # grab
        if self.grab(response) is False:
            return response

        if self.gripper_discard_position(response=response) is False:
            return response

        if self.open_gripper(response) is False:
            return response

        if self.gripper_reset_position(response) is False:
            return response
        
        if self.move_callback(request, response):
            response.message = "Successfully taken piece!"
            response.success = True
        return response

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
            response.message = "Gripper failed to go into reset position."
            response.success = False
        
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
        for atempts in range(6):
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

            time.sleep(2)

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
        trans_right = self.tf_buffer.lookup_transform('link_aruco_fingertip_right', 'base_link')
        trans_target = self.tf_buffer.lookup_transform('BlackPawn3', 'base_link')

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


    def align_callback(self, request, response):
        self.get_logger().info("Align...")

        if self.move_to_square(request.file, request.rank, response) is True:
            response.message = "Aligned to square"
        return response


    
    def move_to_square(self, target_file, target_rank, response):

        file = "File" + target_file.upper()

        if int(target_rank) >= 5:
            file += "B"
        else:
            file += "W"

        rank = "Rank" + target_rank + "W"

        off_by_one_rank_offset = 0.0

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

        self.execute_trajectory(['wrist_extension'], [(index + 0.1) * self.ONE_RANK_DISTANCE - 0.02], duration_sec=5.0)

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
            if self.get_marker_transform(test_frame, "base_link") is not None:
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