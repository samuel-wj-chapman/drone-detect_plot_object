import cv2
import torch  # Assuming PyTorch is used for the model
from torchvision import transforms
from pymavlink import mavutil

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

    # Process the results
    for detection in results.xyxy[0]:  # Assuming results are in xyxy format
        x1, y1, x2, y2, conf, cls = detection
        label = model.names[int(cls)]
        
        # Calculate the centroid
        centroid_x = (x1 + x2) / 2
        centroid_y = (y1 + y2) / 2

        # Print or store the results
        print(f"Label: {label}, Centroid: ({centroid_x}, {centroid_y})")

        # Optionally, draw the detection on the frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Retrieve GPS and altitude data
    msg = mavlink_connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
    if msg:
        latitude = msg.lat / 1e7
        longitude = msg.lon / 1e7
        altitude = msg.alt / 1000.0  # Convert from mm to meters
        print(f"Latitude: {latitude}, Longitude: {longitude}, Altitude: {altitude}m")

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()