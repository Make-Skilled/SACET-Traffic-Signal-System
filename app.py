from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from config import Config
import cv2
import numpy as np
import os
import torch
from PIL import Image
from datetime import datetime

app = Flask(__name__)
app.config.from_object(Config)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///traffic.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize YOLO model
try:
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    # Set model parameters
    yolo_model.conf = 0.25  # Confidence threshold
    yolo_model.classes = [2, 3, 5, 7]  # Class indices for vehicles (car, motorcycle, bus, truck)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    yolo_model = None

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Global variables for signal status and traffic data
signal_status = {
    'north': 'red',
    'south': 'red',
    'east': 'red',
    'west': 'red'
}

wait_times = {
    'north': 0,
    'south': 0,
    'east': 0,
    'west': 0
}

traffic_counts = {
    'north': 0,
    'south': 0,
    'east': 0,
    'west': 0
}

# Timer control status
timer_active = False

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Model for vehicle counts
class VehicleCount(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False)
    direction = db.Column(db.String(10), nullable=False)
    count = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f'<VehicleCount {self.direction}: {self.count} at {self.timestamp}>'

@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('signup'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('signup'))

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash('Registration successful!')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        
        flash('Invalid username or password')
        return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', 
                         signal_status=signal_status,
                         wait_times=wait_times)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/timer_control', methods=['POST'])
@login_required
def timer_control():
    global timer_active
    data = request.get_json()
    action = data.get('action')

    if action == 'start':
        timer_active = True
        return jsonify({
            'status': 'success',
            'message': 'Timer control started'
        })
    elif action == 'stop':
        timer_active = False
        # Reset all signals to red
        for direction in signal_status:
            signal_status[direction] = 'red'
            wait_times[direction] = 0
        
        return jsonify({
            'signal_status': signal_status,
            'wait_times': wait_times
        })
    
    return jsonify({'error': 'Invalid action'}), 400

@app.route('/update_signals', methods=['POST'])
@login_required
def update_signals():
    data = request.get_json()
    current_direction = data.get('current_direction')
    
    if current_direction not in ['north', 'south', 'east', 'west']:
        return jsonify({'error': 'Invalid direction'}), 400
    
    # Update signal status based on current direction
    for direction in signal_status:
        if direction == current_direction:
            signal_status[direction] = 'green'
            wait_times[direction] = 0
        else:
            signal_status[direction] = 'red'
            # Calculate wait time based on position in cycle
            directions = ['north', 'south', 'east', 'west']
            current_index = directions.index(current_direction)
            direction_index = directions.index(direction)
            positions_away = (direction_index - current_index) % len(directions)
            wait_times[direction] = positions_away * 30  # 30 seconds per cycle
    
    return jsonify({
        'signal_status': signal_status,
        'wait_times': wait_times
    })

@app.route('/emergency_stop', methods=['POST'])
@login_required
def emergency_stop():
    global timer_active
    timer_active = False  # Stop timer control if it's running
    
    data = request.get_json()
    allowed_direction = data.get('direction')
    
    if allowed_direction not in ['north', 'south', 'east', 'west']:
        return jsonify({'error': 'Invalid direction'}), 400
    
    # Update signal status
    for direction in signal_status:
        if direction == allowed_direction:
            signal_status[direction] = 'green'
            wait_times[direction] = 0
        else:
            signal_status[direction] = 'red'
            wait_times[direction] = 60  # Set 60 seconds wait time for stopped directions
    
    return jsonify({
        'signal_status': signal_status,
        'wait_times': wait_times
    })

@app.route('/manual_override', methods=['POST'])
@login_required
def manual_override():
    data = request.get_json()
    selected_direction = data.get('direction')
    
    if selected_direction not in ['north', 'south', 'east', 'west']:
        return jsonify({'error': 'Invalid direction'}), 400
    
    # Update signal status
    for direction in signal_status:
        if direction == selected_direction:
            signal_status[direction] = 'green'
            wait_times[direction] = 0
        else:
            signal_status[direction] = 'red'
            wait_times[direction] = 30  # Set 30 seconds wait time for stopped directions
    
    return jsonify({
        'signal_status': signal_status,
        'wait_times': wait_times
    })

@app.route('/analyze_traffic', methods=['POST'])
@login_required
def analyze_traffic():
    if 'photo' not in request.files:
        return jsonify({'error': 'No photo uploaded'}), 400
    
    photo = request.files['photo']
    direction = request.form.get('direction')
    
    if photo.filename == '':
        return jsonify({'error': 'No photo selected'}), 400
    
    if direction not in ['north', 'south', 'east', 'west']:
        return jsonify({'error': 'Invalid direction'}), 400
    
    try:
        # Save the uploaded file
        filename = secure_filename(photo.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        photo.save(filepath)
        
        # Analyze the image using YOLOv5
        vehicle_count = detect_vehicles(filepath)
        traffic_counts[direction] = vehicle_count
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'vehicle_count': vehicle_count,
            'message': 'Analysis completed successfully'
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Error processing image: {str(e)}'
        }), 500

def detect_vehicles(image_path):
    """
    Detect vehicles in an image using YOLOv5.
    Returns the count of vehicles detected.
    """
    if yolo_model is None:
        # Fallback to basic detection if YOLO failed to load
        return fallback_detect_vehicles(image_path)

    try:
        # Read image
        img = Image.open(image_path)
        
        # Perform detection
        results = yolo_model(img)
        
        # Get detections for vehicles only (car, motorcycle, bus, truck)
        vehicle_classes = [2, 3, 5, 7]  # COCO dataset indices for vehicles
        vehicles_detected = results.pred[0]
        
        # Filter detections by confidence and class
        vehicle_count = sum(1 for *_, conf, cls in vehicles_detected.tolist() 
                          if int(cls) in vehicle_classes and conf > yolo_model.conf)
        
        return vehicle_count

    except Exception as e:
        print(f"Error in vehicle detection: {e}")
        return fallback_detect_vehicles(image_path)

def fallback_detect_vehicles(image_path):
    """
    Fallback method using basic OpenCV detection if YOLO fails.
    """
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')
        vehicles = car_cascade.detectMultiScale(gray, 1.1, 3)
        return len(vehicles)
    except Exception as e:
        print(f"Error in fallback detection: {e}")
        return 0

@app.route('/start_auto_control', methods=['POST'])
@login_required
def start_auto_control():
    data = request.json
    vehicle_counts = data.get('vehicle_counts', {})
    timestamp_str = data.get('timestamp')

    try:
        # Parse timestamp
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

        # Store each direction's count in the database
        for direction, count in vehicle_counts.items():
            vehicle_count = VehicleCount(
                timestamp=timestamp,
                direction=direction,
                count=count
            )
            db.session.add(vehicle_count)
        db.session.commit()

        # Find direction with highest traffic
        max_direction = max(vehicle_counts.items(), key=lambda x: x[1])[0]
        
        # Update signal status based on traffic density
        for direction in signal_status:
            if direction == max_direction:
                signal_status[direction] = 'green'
                wait_times[direction] = 0
            else:
                signal_status[direction] = 'red'
                # Calculate wait time based on vehicle count difference
                wait_times[direction] = min(60, max(30, vehicle_counts[max_direction] - vehicle_counts[direction]) * 10)

        return jsonify({
            'status': 'success',
            'signal_status': signal_status,
            'wait_times': wait_times
        })
    except Exception as e:
        db.session.rollback()
        print(f"Error in start_auto_control: {str(e)}")  # Add debug logging
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_total_vehicles', methods=['GET'])
def get_total_vehicles():
    try:
        # Get the sum of all vehicle counts from the database
        total = db.session.query(db.func.sum(VehicleCount.count)).scalar() or 0
        return jsonify({'total_vehicles': total})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True) 