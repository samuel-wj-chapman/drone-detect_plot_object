import cv2
import torch  # Assuming PyTorch is used for the model
from torchvision import transforms
from pymavlink import mavutil
from translate import calculate_gps_from_detection


heading = 90  # add later correct method for heading


# Load your YOLOv8 model (assuming it's a PyTorch model)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Example for YOLOv5, replace with YOLOv8

# Initialize the video capture
cap = cv2.VideoCapture(0)  # 0 is usually the default camera

# Define a transformation for the input image
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add any other necessary transformations
])

# Connect to the flight controller
mavlink_connection = mavutil.mavlink_connection('/dev/ttyTHS1', baud=57600)  # Adjust port and baud rate as needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_tensor = transform(frame).unsqueeze(0)  # Add batch dimension

    # Run the model
    results = model(input_tensor)
    msg = mavlink_connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
    if msg:
        drone_lat = msg.lat / 1e7
        drone_lon = msg.lon / 1e7
        drone_alt = msg.alt / 1000.0  # Convert from mm to meters

    # Process the results
    for detection in results.xyxy[0]:  # Assuming results are in xyxy format

        x1, y1, x2, y2, conf, cls = detection
        if label not in ['person', 'car', 'truck', 'motorbike', 'boat']:
            break
        label = model.names[int(cls)]
        
        # Calculate the centroid
        centroid_x = (x1 + x2) / 2
        centroid_y = (y1 + y2) / 2

        detection_lat, detection_lon = calculate_gps_from_detection(
            drone_lat, drone_lon, drone_alt, heading, 90,  # Assuming FOV is 90 degrees
            frame.shape[1], frame.shape[0], centroid_x, centroid_y
        )
    # Retrieve GPS and altitude data

    # Display the frame
    

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()