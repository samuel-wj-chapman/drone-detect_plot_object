import cv2
import torch
from flask_socketio import SocketIO
from translate import calculate_gps_from_detection
import numpy as np

# Initialize SocketIO
socketio = SocketIO(message_queue='redis://')

# Load your YOLOv8 model (assuming it's a PyTorch model)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Example for YOLOv5, replace with YOLOv8

# Initialize the video capture with a video file
cap = cv2.VideoCapture('path/to/your/video.mp4')  # Replace with your video file path

# Define a transformation for the input image
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add any other necessary transformations
])

# Simulate initial GPS coordinates
drone_lat = 37.7749
drone_lon = -122.4194
drone_alt = 100  # Example altitude in meters
heading = 90  # Example heading in degrees
fov = 90  # Example field of view in degrees

# Simulate drone movement
def simulate_drone_movement(lat, lon, step=0.0001):
    # Simulate a simple linear movement
    return lat + step, lon + step

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_tensor = transform(frame).unsqueeze(0)  # Add batch dimension

    # Run the model
    results = model(input_tensor)

    # Simulate drone movement
    drone_lat, drone_lon = simulate_drone_movement(drone_lat, drone_lon)

    # Process the results
    for detection in results.xyxy[0]:  # Assuming results are in xyxy format
        x1, y1, x2, y2, conf, cls = detection
        label = model.names[int(cls)]
        if label not in ['person', 'car', 'truck', 'motorbike', 'boat']:
            continue

        # Calculate the centroid
        centroid_x = (x1 + x2) / 2
        centroid_y = (y1 + y2) / 2

        detection_lat, detection_lon = calculate_gps_from_detection(
            drone_lat, drone_lon, drone_alt, heading, fov,
            frame.shape[1], frame.shape[0], centroid_x, centroid_y
        )
        socketio.emit('detection_update', {'lat': detection_lat, 'lng': detection_lon})

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()