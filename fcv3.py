import cv2
import numpy as np

# Load the cat image (ensure the image is in the same folder or provide a full path)
cat_image = cv2.imread('rohit.png')  # Make sure you have the 'rohit.png' image

# Check OpenCV version
print(cv2.__version__)

# Initialize the ArUco dictionary and parameters using getPredefinedDictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()  # Updated for OpenCV 4.10.0

# Create an ArUco detector object (new approach in OpenCV 4.10.0)
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Define the size of the ArUco marker in meters
marker_size = 0.05  # 5 cm

# Load the camera calibration parameters (intrinsic matrix and distortion coefficients)
# These values should come from camera calibration. Below is an example with dummy values.
# You must replace them with your own calibrated values.
camera_matrix = np.array([[1000, 0, 320],
                          [0, 1000, 240],
                          [0, 0, 1]], dtype=float)
dist_coeffs = np.zeros((5, 1))  # Assuming no lens distortion

# Function to overlay the cat image
def overlay_image(frame, cat_img, corners, height_offset=0.05):
    # Define the points of the floating plane above the ArUco marker
    h, w = cat_img.shape[:2]
    
    # Define the 3D points of where the cat image should appear
    floating_corners = np.float32([
        [-0.025, -0.025, height_offset],  # Bottom-left (floating 5 cm above)
        [0.025, -0.025, height_offset],   # Bottom-right
        [0.025, 0.025, height_offset],    # Top-right
        [-0.025, 0.025, height_offset],   # Top-left
    ])

    # Project the 3D points to the image plane
    imgpts, _ = cv2.projectPoints(floating_corners, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # Convert the destination points (imgpts) to float32
    dst_pts = np.float32(imgpts)

    # Define the source points as the corners of the cat image
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Get the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Warp the cat image onto the frame
    warped_cat = cv2.warpPerspective(cat_img, M, (frame.shape[1], frame.shape[0]))

    # Create a mask for the cat image
    mask = np.zeros_like(frame, dtype=np.uint8)
    mask = cv2.fillConvexPoly(mask, np.int32(dst_pts), (255, 255, 255))
    
    # Bitwise operations to overlay the cat image
    frame_bg = cv2.bitwise_and(frame, cv2.bitwise_not(mask))  # Remove the region where the cat will be placed
    frame_fg = cv2.bitwise_and(warped_cat, mask)  # Extract the warped cat image

    # Combine the background and the foreground
    frame = cv2.add(frame_bg, frame_fg)

    return frame

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the image using the ArUcoDetector object
    corners, ids, rejected = aruco_detector.detectMarkers(gray)

    # If an ArUco marker is detected
    if ids is not None:
        # Estimate the pose of the ArUco marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
        
        # Loop through all detected markers
        for rvec, tvec in zip(rvecs, tvecs):
            # Draw the detected marker
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Overlay the cat image above the ArUco marker
            frame = overlay_image(frame, cat_image, corners[0], height_offset=0.05)

    # Display the frame
    cv2.imshow('Floating Cat Image on ArUco Marker', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
