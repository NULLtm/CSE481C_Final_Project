import rclpy
from rclpy.node import Node
from interfaces.srv import Align
from flask import Flask, render_template, jsonify
import threading
import time

app = Flask(__name__)

class RobotClient(Node):
    def __init__(self):
        super().__init__('flask_robot_client')
        self.client = self.create_client(Align, 'align')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service /align not available, waiting...')
        self.get_logger().info('Service /align is ready!')

robot_node = None

def ros_spin_thread():
    global robot_node
    rclpy.init()
    robot_node = RobotClient()
    rclpy.spin(robot_node)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/move/<int:rank>', methods=['POST'])
def move_to_rank(rank):
    global robot_node
    if robot_node is None or not robot_node.client.service_is_ready():
        return jsonify({"status": "error", "message": "ROS 2 node or service not ready"}), 500

    req = Align.Request()
    req.pos = f"Rank{rank}"
    
    # We are in a Flask request thread, so we can wait for the future here.
    future = robot_node.client.call_async(req)
    
    # Wait for result (blocking this specific request thread)
    while not future.done():
        time.sleep(0.1)
    
    try:
        response = future.result()
        return jsonify({"status": "success", "message": response.message})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Start ROS 2 in a separate thread
    thread = threading.Thread(target=ros_spin_thread, daemon=True)
    thread.start()
    
    # Give it a moment to initialize
    time.sleep(2)
    
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)