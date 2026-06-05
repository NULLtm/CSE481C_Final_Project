import rclpy
from rclpy.node import Node
from flask import Flask, render_template, jsonify, request
import threading
import time

# TODO: Replace these placeholders with your actual ROS 2 service types
from interfaces.srv import Move, Align
from std_srvs.srv import Trigger

app = Flask(__name__)

class RobotClient(Node):
    def __init__(self):
        super().__init__('flask_robot_client')
        
        # 1. Setup Reset Client
        self.reset_client = self.create_client(Trigger, '/chess/reset')
        # 2. Setup Align Client
        self.align_client = self.create_client(Align, '/chess/align')
        # 3. Setup Move Client
        self.move_client = self.create_client(Move, '/chess/move')
        # 3. Setup Take Client
        self.take_client = self.create_client(Move, '/chess/take')

        self.get_logger().info('Waiting for ROS 2 services to become available...')
        self.reset_client.wait_for_service()
        self.align_client.wait_for_service()
        self.move_client.wait_for_service()
        self.take_client.wait_for_service()
        self.get_logger().info('All chess services are ready!')

robot_node = None

def ros_spin_thread():
    global robot_node
    rclpy.init()
    robot_node = RobotClient()
    rclpy.spin(robot_node)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reset', methods=['POST'])
def reset_robot():
    global robot_node
    if robot_node is None or not robot_node.reset_client.service_is_ready():
        return jsonify({"status": "error", "message": "ROS 2 node or service not ready"}), 500

    req = Trigger.Request()
    # TODO: Populate any necessary fields for your Reset request here
    
    future = robot_node.reset_client.call_async(req)
    
    while not future.done():
        time.sleep(0.1)
    
    try:
        response = future.result()
        # TODO: Adjust to read your specific response attributes
        return jsonify({"status": "success", "message": "Reset executed successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/align', methods=['POST'])
def align_robot():
    global robot_node
    if robot_node is None or not robot_node.align_client.service_is_ready():
        return jsonify({"status": "error", "message": "ROS 2 node or service not ready"}), 500

    # Extract JSON payload from frontend
    data = request.get_json()
    target_file = data.get('file')
    target_rank = data.get('rank')

    req = Align.Request()
    # TODO: Map target_file and target_rank to your Request object
    # e.g., req.file = target_file
    # e.g., req.rank = int(target_rank)

    req.file = target_file
    req.rank = target_rank
    
    future = robot_node.align_client.call_async(req)
    
    while not future.done():
        time.sleep(0.1)
    
    try:
        response = future.result()
        return jsonify({"status": "success", "message": f"Aligned to {target_file}{target_rank}."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/move', methods=['POST'])
def move_robot():
    global robot_node
    if robot_node is None or not robot_node.move_client.service_is_ready():
        return jsonify({"status": "error", "message": "ROS 2 node or service not ready"}), 500

    # Extract JSON payload from frontend
    data = request.get_json()
    start_file = data.get('start_file')
    start_rank = data.get('start_rank')
    end_file = data.get('end_file')
    end_rank = data.get('end_rank')

    req = Move.Request()
    # TODO: Map start/end files and ranks to your Request object
    # e.g., req.start_pos = f"{start_file}{start_rank}"

    req.start_file = start_file
    req.end_file = end_file
    req.start_rank = start_rank
    req.end_rank = end_rank
    
    future = robot_node.move_client.call_async(req)
    
    while not future.done():
        time.sleep(0.1)
    
    try:
        response = future.result()
        return jsonify({"status": "success", "message": f"Moved from {start_file}{start_rank} to {end_file}{end_rank}."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/take', methods=['POST'])
def take_robot():
    global robot_node
    if robot_node is None or not robot_node.take_client.service_is_ready():
        return jsonify({"status": "error", "message": "ROS 2 node or service not ready"}), 500

    # Extract JSON payload from frontend
    data = request.get_json()
    start_file = data.get('start_file')
    start_rank = data.get('start_rank')
    end_file = data.get('end_file')
    end_rank = data.get('end_rank')

    req = Move.Request()
    # TODO: Map start/end files and ranks to your Request object
    # e.g., req.start_pos = f"{start_file}{start_rank}"

    req.start_file = start_file
    req.end_file = end_file
    req.start_rank = start_rank
    req.end_rank = end_rank
    
    future = robot_node.take_client.call_async(req)
    
    while not future.done():
        time.sleep(0.1)
    
    try:
        response = future.result()
        return jsonify({"status": "success", "message": f"Took from {start_file}{start_rank} to {end_file}{end_rank}."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Start ROS 2 in a separate thread
    thread = threading.Thread(target=ros_spin_thread, daemon=True)
    thread.start()
    
    # Give it a moment to initialize
    time.sleep(2)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)