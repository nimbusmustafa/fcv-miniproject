import cv2
import numpy as np
from stl import mesh

# Function to project and draw a 3D model on the ArUco marker
def render_3d_model_on_marker(frame, stl_model, rvec, tvec, camera_matrix, dist_coeffs, scale_factor=0.05):
    # Create 3D points (vertices) from the STL model and scale them
    object_points = np.array([vertex for facet in stl_model.vectors for vertex in facet], dtype=np.float32) * scale_factor
    
    # Project the 3D points to 2D image points
    imgpts, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    
    # Draw the projected points as polygons
    imgpts = np.int32(imgpts).reshape(-1, 2)
    
    # Draw the triangles on the image frame
    for i in range(0, len(imgpts), 3):
        cv2.polylines(frame, [imgpts[i:i+3]], isClosed=True, color=(0, 255, 0), thickness=2)

# Load your STL model (replace with the correct path to your .stl file)
stl_model = mesh.Mesh.from_file('spider.stl')  # Replace this with the path to your .stl file

# Initialize OpenCV ArUco dictionary and detector parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# Camera calibration parameters (intrinsic matrix and distortion coefficients)
# Replace these with your calibrated values
camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=float)
dist_coeffs = np.zeros((5, 1))  # Assuming no lens distortion

# Marker size in meters
marker_size = 0.01  # 1 cm marker size

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get frame width, height, and frames per second from the webcam
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object to save the recording
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 format
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the image
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # Estimate the pose of the ArUco marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
        
        for rvec, tvec in zip(rvecs, tvecs):
            # Draw the detected marker on the frame
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Render the 3D model on top of the detected ArUco marker with a reduced size
            render_3d_model_on_marker(frame, stl_model, rvec, tvec, camera_matrix, dist_coeffs, scale_factor=0.0001)
            print("3D model is being rendered on the webcam feed.")

    # Write the current frame with 3D model into the video file
    out.write(frame)

    # Display the webcam feed with the 3D model
    cv2.imshow('Webcam Feed', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam, video writer, and close OpenCV windows after exit
cap.release()
out.release()  # Save the video file
cv2.destroyAllWindows()
