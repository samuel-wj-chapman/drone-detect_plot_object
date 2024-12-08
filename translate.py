


import math
import cv2

def undistort_fisheye_image(image, camera_matrix, dist_coeffs):
    """
    Undistort a fisheye image using the camera matrix and distortion coefficients.

    :param image: The input fisheye image
    :param camera_matrix: Camera matrix obtained from calibration
    :param dist_coeffs: Distortion coefficients obtained from calibration
    :return: Undistorted image
    """
    # Get the optimal new camera matrix
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    # Undistort the image
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Crop the image based on the ROI
    x, y, w, h = roi
    undistorted_image = undistorted_image[y:y+h, x:x+w]

    return undistorted_image


def calculate_gps_from_detection(drone_lat, drone_lon, drone_alt, heading, fov, image_width, image_height, detection_x, detection_y):
    """
    Calculate GPS coordinates from a detection on an image.

    :param drone_lat: Latitude of the drone
    :param drone_lon: Longitude of the drone
    :param drone_alt: Altitude of the drone
    :param heading: Heading of the drone in degrees
    :param fov: Field of view of the camera in degrees
    :param image_width: Width of the image in pixels
    :param image_height: Height of the image in pixels
    :param detection_x: X coordinate of the detection in the image
    :param detection_y: Y coordinate of the detection in the image
    :return: Tuple of (latitude, longitude) of the detection
    """
    # Convert heading and FOV to radians
    heading_rad = math.radians(heading)
    fov_rad = math.radians(fov)

    # Calculate the angle of the detection relative to the center of the image
    angle_x = (detection_x - image_width / 2) / image_width * fov_rad
    angle_y = (detection_y - image_height / 2) / image_height * fov_rad

    # Calculate the distance from the drone to the detection point
    distance = drone_alt / math.tan(fov_rad / 2)

    # Calculate the GPS offset
    delta_lat = distance * math.cos(heading_rad + angle_x) / 111111  # Approx. meters per degree latitude
    delta_lon = distance * math.sin(heading_rad + angle_y) / (111111 * math.cos(math.radians(drone_lat)))

    # Calculate the GPS coordinates of the detection
    detection_lat = drone_lat + delta_lat
    detection_lon = drone_lon + delta_lon

    return detection_lat, detection_lon

# Example usage
drone_lat = 37.7749  # Example latitude
drone_lon = -122.4194  # Example longitude
drone_alt = 100  # Example altitude in meters
heading = 90  # Example heading in degrees
fov = 90  # Example field of view in degrees
image_width = 1920  # Example image width in pixels
image_height = 1080  # Example image height in pixels
detection_x = 960  # Example detection x coordinate
detection_y = 540  # Example detection y coordinate

detection_lat, detection_lon = calculate_gps_from_detection(
    drone_lat, drone_lon, drone_alt, heading, fov, image_width, image_height, detection_x, detection_y
)

print(f"Detection GPS Coordinates: Latitude = {detection_lat}, Longitude = {detection_lon}")