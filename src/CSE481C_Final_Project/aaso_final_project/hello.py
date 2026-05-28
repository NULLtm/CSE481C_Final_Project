import rclpy
from std_srvs.srv import Trigger
import hello_helpers.hello_misc as hm

class ArucoBaseAlignNode(hm.HelloNode):
    
    def __init__(self):
        hm.HelloNode.__init__(self)
        
        # --- CONFIGURATION ---
        self.aruco_frame = 'aruco_marker_0' 
        self.robot_frame = 'base_link'      
        self.desired_distance_offset = 0.50 

    def main(self):
        # 1. Initialize HelloNode (This automatically sets up TF listeners and joint state subscribers!)
        hm.HelloNode.main(
            self, 
            'aruco_base_align_node', 
            'aruco_base_align_node', 
            wait_for_first_pointcloud=False
        )

        # 2. Setup Custom Trigger Service
        self.trigger_service = self.create_service(
            Trigger,
            'align_to_aruco',
            self.align_base_callback
        )

        self.get_logger().info('Ready! Call /align_to_aruco to align to the marker.')

        # 3. Spin the node
        try:
            rclpy.spin(self)
        except KeyboardInterrupt:
            pass
        finally:
            self.destroy_node()
            rclpy.try_shutdown()

    # =========================================================
    # Callbacks
    # =========================================================

    def align_base_callback(self, request, response):
        """Triggered via service call. Looks up TF and moves the base."""
        self.get_logger().info('Trigger received: Looking for ArUco marker...')

        # Failsafe 1: Check Joint States (Populated automatically by HelloNode)
        if self.joint_state is None:
            response.success = False
            response.message = 'No joint states available yet.'
            self.get_logger().error(response.message)
            return response

        # Failsafe 2: Use HelloNode's built-in TF lookup
        try:
            transform_stamped = self.get_tf(self.robot_frame, self.aruco_frame)
            if transform_stamped is None:
                raise ValueError("Transform is unavailable.")
        except Exception as ex:
            response.success = False
            response.message = f'Could not find ArUco marker "{self.aruco_frame}". Is it in view?'
            self.get_logger().error(f'{response.message} Details: {ex}')
            return response

        # --- Calculate the Math ---
        # Get forward distance from base_link to marker
        distance_to_marker = transform_stamped.transform.translation.x
        move_distance = distance_to_marker - self.desired_distance_offset

        # Build joint dictionary from HelloNode's cached joint_state to find current base pos
        joint_dict = dict(zip(
            self.joint_state.name,
            self.joint_state.position
        ))
        current_base = joint_dict.get("translate_mobile_base", 0.0)
        
        # Calculate final joint target
        target_base = current_base + move_distance

        self.get_logger().info(
            f'Marker is {distance_to_marker:.3f}m away. '
            f'Moving base by {move_distance:.3f}m to reach {self.desired_distance_offset}m offset.'
        )

        # Use HelloNode's built-in motion function asynchronously
        self.move_to_pose({'translate_mobile_base': target_base}, blocking=False)

        # Return success immediately
        response.success = True
        response.message = f'Aligning to marker. Moving {move_distance:.3f}m.'
        return response

if __name__ == '__main__':
    node = ArucoBaseAlignNode()
    node.main()