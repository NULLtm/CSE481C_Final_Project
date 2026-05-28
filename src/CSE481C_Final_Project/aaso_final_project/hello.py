import rclpy
from std_srvs.srv import Trigger
import hello_helpers.hello_misc as hm

class HelloTriggerNode(hm.HelloNode):
    
    def __init__(self):
        # Initialize the base Python class, but do NOT initialize ROS2 yet.
        hm.HelloNode.__init__(self)

    def main(self):
        # 1. Initialize the HelloNode ROS2 architecture
        # This handles rclpy.init(), names the node, and sets up joint state caching.
        hm.HelloNode.main(
            self, 
            'hello_trigger_node', 
            'hello_trigger_node', 
            wait_for_first_pointcloud=False
        )

        # 2. Setup your custom ROS2 interfaces
        self.hello_service = self.create_service(
            Trigger,
            'Hello',
            self.hello_callback
        )

        self.get_logger().info('HelloTriggerNode is ready. Call the /Hello service!')

        # 3. Spin the node to process callbacks
        try:
            rclpy.spin(self)
        except KeyboardInterrupt:
            pass
        finally:
            # Clean up on exit
            self.destroy_node()
            rclpy.try_shutdown()

    def hello_callback(self, request, response):
        """Callback for the /Hello Trigger service."""
        
        # Print to the console (ROS2 logger)
        self.get_logger().info('Hello')
        
        # Populate the Trigger response
        response.success = True
        response.message = 'Successfully printed Hello to the console.'
        
        return response

if __name__ == '__main__':
    node = HelloTriggerNode()
    node.main()
