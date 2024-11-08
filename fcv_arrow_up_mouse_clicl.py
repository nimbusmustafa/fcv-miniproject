import cv2
import numpy as np

# Global variable to store corners of the detected ArUco marker
detected_corners = None

# Mouse click event handler
def on_mouse(event, x, y, flags, param):
    global detected_corners
    if event == cv2.EVENT_LBUTTONDOWN:
        if detected_corners is not None:
            # Check if the click is inside the detected ArUco marker area
            corners = np.array(detected_corners).reshape((4, 2))
            if cv2.pointPolygonTest(corners, (x, y), False) >= 0:
                print("Button clicked!")

# Function to overlay an arrow on the ArUco marker
def overlay_arrow_on_marker(frame, corners, arrow_img):
    # Get the size of the arrow image
    arrow_h, arrow_w = arrow_img.shape[:2]
    
    # Get the corners of the marker (top-left, top-right, bottom-right, bottom-left)
    pts_dst = np.array(corners, dtype="float32")
    
    # Define the source points from the arrow image (corners of the image)
    pts_src = np.array([[0, 0], [arrow_w, 0], [arrow_w, arrow_h], [0, arrow_h]], dtype="float32")
    
    # Get the transformation matrix to warp the arrow image onto the marker
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    
    # Warp the arrow image onto the marker area
    warped_arrow = cv2.warpPerspective(arrow_img, M, (frame.shape[1], frame.shape[0]))

    # Check if the arrow image has an alpha channel (transparency)
    if warped_arrow.shape[2] == 4:  # 4 channels (RGBA)
        # Split the arrow image into RGB and Alpha channels
        arrow_rgb = warped_arrow[:, :, :3]  # RGB channels
        alpha_channel = warped_arrow[:, :, 3]  # Alpha channel

        # Create a mask from the alpha channel and inverse mask
        mask = cv2.merge([alpha_channel, alpha_channel, alpha_channel])
        mask_inv = cv2.bitwise_not(mask)

        # Black-out the area of the arrow in the frame using the mask_inv
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv[:, :, 0])

        # Extract only the arrow region from the warped arrow image
        arrow_fg = cv2.bitwise_and(arrow_rgb, arrow_rgb, mask=mask[:, :, 0])

        # Combine the frame and the arrow image
        frame = cv2.add(frame_bg, arrow_fg)
    else:
        # If the arrow image doesn't have an alpha channel, just overlay it
        frame = cv2.add(frame, warped_arrow)

    return frame

# Load the up arrow image (ensure it's in the same directory or provide a full path)
arrow_img = cv2.imread('arrowup.png', cv2.IMREAD_UNCHANGED)

# Ensure arrow image is loaded correctly
if arrow_img is None:
    raise Exception("Arrow image not found! Ensure up_arrow.png is in the correct path.")

# Initialize OpenCV ArUco dictionary and detector parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
parameters = cv2.aruco.DetectorParameters()

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set up mouse callback for click detection
cv2.namedWindow('Webcam Feed')
cv2.setMouseCallback('Webcam Feed', on_mouse)

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
        # Draw the detected markers on the frame
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Overlay the arrow on each detected marker
        for marker_corners in corners:
            frame = overlay_arrow_on_marker(frame, marker_corners[0], arrow_img)
            detected_corners = marker_corners[0]  # Save the corners for click detection
    
    # Display the webcam feed with the arrow overlay
    cv2.imshow('Webcam Feed', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows after exit
cap.release()
cv2.destroyAllWindows()
