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
        # 4. Setup Take Client
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
    
    future = robot_node.reset_client.call_async(req)
    
    while not future.done():
        time.sleep(0.1)
    
    try:
        response = future.result()
        return jsonify({"status": "success", "message": "Reset executed successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/align', methods=['POST'])
def align_robot():
    global robot_node
    if robot_node is None or not robot_node.align_client.service_is_ready():
        return jsonify({"status": "error", "message": "ROS 2 node or service not ready"}), 500

    data = request.get_json()
    target_file = data.get('file')
    target_rank = data.get('rank')

    req = Align.Request()
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

    data = request.get_json()
    start_piece = data.get('start_piece')
    start_file = data.get('start_file')
    start_rank = data.get('start_rank')
    end_file = data.get('end_file')
    end_rank = data.get('end_rank')

    req = Move.Request()
    req.start_piece = start_piece
    req.end_piece = "" # Explicitly left empty for move
    req.start_file = start_file
    req.end_file = end_file
    req.start_rank = start_rank
    req.end_rank = end_rank
    
    future = robot_node.move_client.call_async(req)
    
    while not future.done():
        time.sleep(0.1)
    
    try:
        response = future.result()
        return jsonify({"status": "success", "message": f"Moved {start_piece} from {start_file}{start_rank} to {end_file}{end_rank}."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/take', methods=['POST'])
def take_robot():
    global robot_node
    if robot_node is None or not robot_node.take_client.service_is_ready():
        return jsonify({"status": "error", "message": "ROS 2 node or service not ready"}), 500

    data = request.get_json()
    start_piece = data.get('start_piece')
    end_piece = data.get('end_piece')
    start_file = data.get('start_file')
    start_rank = data.get('start_rank')
    end_file = data.get('end_file')
    end_rank = data.get('end_rank')

    req = Move.Request()
    req.start_piece = start_piece
    req.end_piece = end_piece
    req.start_file = start_file
    req.end_file = end_file
    req.start_rank = start_rank
    req.end_rank = end_rank
    
    future = robot_node.take_client.call_async(req)
    
    while not future.done():
        time.sleep(0.1)
    
    try:
        response = future.result()
        return jsonify({"status": "success", "message": f"Piece {start_piece} at {start_file}{start_rank} took {end_piece} at {end_file}{end_rank}."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    thread = threading.Thread(target=ros_spin_thread, daemon=True)
    thread.start()
    
    time.sleep(2)
    
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)