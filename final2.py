import cv2
import numpy as np
import time

# Global variables to track marker disappearance timing
marker_timers = {0: None, 1: None, 2: None}  # Track last seen time for up, down, and edit arrows
marker_visibility = {0: True, 1: True, 2: True}  # Track whether the markers are visible
disappearance_threshold = 2.0  # Time in seconds before triggering video change or edit change

# Load the up, down, and edit arrow images
up_arrow_img = cv2.imread('arrowup.png', cv2.IMREAD_UNCHANGED)
down_arrow_img = cv2.imread('arrowdown.png', cv2.IMREAD_UNCHANGED)
edit_arrow_img = cv2.imread('editt.png', cv2.IMREAD_UNCHANGED)

# Load the videos
video_files = ['video1.mp4', 'video2.mp4', 'output.mp4', 'rohit.mp4']
current_video_index = 0
ProjVid_Cap = cv2.VideoCapture(video_files[current_video_index])

# Video effect states (Grayscale → Canny Edge → Gaussian Blur → Negative)
video_effects = ["normal", "grayscale", "canny", "blur", "negative"]
current_effect_index = 0

# Function to switch the video
def switch_video(direction):
    global current_video_index, ProjVid_Cap
    current_video_index = (current_video_index + direction) % len(video_files)
    ProjVid_Cap = cv2.VideoCapture(video_files[current_video_index])
    print(f"Switched to video {current_video_index + 1}")

# Function to apply video effect
def apply_video_effect(video_frame):
    global current_effect_index
    effect = video_effects[current_effect_index]

    if effect == "grayscale":
        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for consistency
    elif effect == "canny":
        video_frame = cv2.Canny(video_frame, 100, 200)
        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for consistency
    elif effect == "blur":
        video_frame = cv2.GaussianBlur(video_frame, (15, 15), 0)
    elif effect == "negative":
        video_frame = cv2.bitwise_not(video_frame)
    # "normal" has no effect
    return video_frame

# Function to overlay an arrow on the ArUco marker
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

    # Ensure warped_arrow has the same number of channels as frame
    if len(warped_arrow.shape) == 2:  # Single channel (grayscale)
        warped_arrow = cv2.cvtColor(warped_arrow, cv2.COLOR_GRAY2BGR)
    elif warped_arrow.shape[2] == 4:  # RGBA (4 channels)
        # Split the arrow image into RGB and Alpha channels
        arrow_rgb = warped_arrow[:, :, :3]  # RGB channels
        alpha_channel = warped_arrow[:, :, 3]  # Alpha channel

        # Create a mask from the alpha channel and inverse mask
        mask = cv2.merge([alpha_channel, alpha_channel, alpha_channel])
        mask_inv = cv2.bitwise_not(mask)

        # Ensure mask and mask_inv are the same size as the frame
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        mask_inv = cv2.resize(mask_inv, (frame.shape[1], frame.shape[0]))

        # Black-out the area of the arrow in the frame using the mask_inv
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv[:, :, 0])

        # Extract only the arrow region from the warped arrow image
        arrow_fg = cv2.bitwise_and(arrow_rgb, arrow_rgb, mask=mask[:, :, 0])

        # Ensure both background and foreground have the same size
        arrow_fg = cv2.resize(arrow_fg, (frame.shape[1], frame.shape[0]))
        frame_bg = cv2.resize(frame_bg, (frame.shape[1], frame.shape[0]))

        # Combine the frame and the arrow image
        frame = cv2.add(frame_bg, arrow_fg)
    else:
        # If the arrow image doesn't have an alpha channel, resize it to match the frame dimensions
        warped_arrow = cv2.resize(warped_arrow, (frame.shape[1], frame.shape[0]))
        frame = cv2.add(frame, warped_arrow)

    return frame


# Function to overlay video on four ArUco markers
def overlay_video_on_markers(frame, marker_centers, video_frame):
    if len(marker_centers) != 4:
        return frame  # If not exactly four markers, return the original frame

    # Define the source points from the video (top-left, top-right, bottom-right, bottom-left)
    h, w = video_frame.shape[:2]
    pts_src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")

    # Use the marker centers to define the corners of the video
    pts_dst = np.array(marker_centers, dtype="float32")

    # Get the perspective transform matrix and warp the video onto the marker area
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped_video = cv2.warpPerspective(video_frame, M, (frame.shape[1], frame.shape[0]))

    # Create a mask for the video and blend it into the frame
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(pts_dst), (255, 255, 255))
    frame_bg = cv2.bitwise_and(frame, cv2.bitwise_not(mask))
    frame_fg = cv2.bitwise_and(warped_video, mask)
    frame = cv2.add(frame_bg, frame_fg)

    return frame

# Function to overlay the video onto the detected ArUco markers
def overlay_video_on_markerss(frame, marker_centers, video_frame):
    # Ensure that exactly four markers (0, 1, 2, 3) are detected
    if set(marker_centers.keys()) != {0, 1, 2, 3}:
        return frame  # If not all required markers are detected, return the original frame

    # Get dimensions of the video frame to be overlaid
    h, w = video_frame.shape[:2]

    # Define source points from the video (top-left, top-right, bottom-right, bottom-left)
    pts_src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")

    # Define destination points from marker centers, in the correct order
    pts_dst = np.array([
        marker_centers[0],  # ID 0: top-left
        marker_centers[1],  # ID 1: top-right
        marker_centers[2],  # ID 2: bottom-right
        marker_centers[3]   # ID 3: bottom-left
    ], dtype="float32")

    # Compute the perspective transform matrix and apply it to the video frame
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped_video = cv2.warpPerspective(video_frame, M, (frame.shape[1], frame.shape[0]))

    # Create a mask for the overlay area
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(pts_dst), (255, 255, 255))

    # Combine the overlay with the background
    frame_bg = cv2.bitwise_and(frame, cv2.bitwise_not(mask))
    frame_fg = cv2.bitwise_and(warped_video, mask)
    frame = cv2.add(frame_bg, frame_fg)

    return frame

# Initialize OpenCV ArUco dictionary and detector parameters
aruco_dict_6x6 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
aruco_dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# Open the webcam
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    retProjVid, ProjVid_Frame = ProjVid_Cap.read()
    if not retProjVid:
        ProjVid_Cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        retProjVid, ProjVid_Frame = ProjVid_Cap.read()

    # Ensure the video frame is valid and apply the current video effect
    if ProjVid_Frame is not None:
        ProjVid_Frame = apply_video_effect(ProjVid_Frame)
    else:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers for both the video overlay (6x6 markers) and buttons (4x4 markers)
    corners_6x6, ids_6x6, _ = cv2.aruco.detectMarkers(gray, aruco_dict_6x6, parameters=parameters)
    corners_4x4, ids_4x4, _ = cv2.aruco.detectMarkers(gray, aruco_dict_4x4, parameters=parameters)

    # Handle marker disappearance and video/effect change for 4x4 markers
    current_time = time.time()
    found_markers = set()

    # Overlay arrow images on 4x4 markers and track visibility
    if ids_4x4 is not None:
        for marker_corners, marker_id in zip(corners_4x4, ids_4x4):
            found_markers.add(marker_id[0])
            if marker_id == 0:  # Up arrow marker (ID 0)
                frame = overlay_arrow_on_marker(frame, marker_corners[0], up_arrow_img)
                marker_timers[0] = current_time
                marker_visibility[0] = True
            elif marker_id == 1:  # Down arrow marker (ID 1)
                frame = overlay_arrow_on_marker(frame, marker_corners[0], down_arrow_img)
                marker_timers[1] = current_time
                marker_visibility[1] = True
            elif marker_id == 2:  # Edit arrow marker (ID 2)
                frame = overlay_arrow_on_marker(frame, marker_corners[0], edit_arrow_img)
                marker_timers[2] = current_time
                marker_visibility[2] = True

    for marker_id in marker_timers:
        if marker_id not in found_markers:
            if marker_timers[marker_id] is not None:
                elapsed_time = current_time - marker_timers[marker_id]
                if elapsed_time >= disappearance_threshold:
                    if marker_id == 0 and marker_visibility[0]:
                        print("Next video (Up arrow hidden)!")
                        switch_video(1)
                        marker_visibility[0] = False
                    elif marker_id == 1 and marker_visibility[1]:
                        print("Previous video (Down arrow hidden)!")
                        switch_video(-1)
                        marker_visibility[1] = False
                    elif marker_id == 2 and marker_visibility[2]:
                        print("Change video effect (Edit arrow hidden)!")
                        current_effect_index = (current_effect_index + 1) % len(video_effects)
                        marker_visibility[2] = False
    marker_centers = {}
    if ids_6x6 is not None:
        for marker_corners, marker_id in zip(corners_6x6, ids_6x6.flatten()):
            if marker_id in {0, 1, 2, 3}:
                marker_centers[marker_id] = np.mean(marker_corners[0], axis=0)

    # Overlay video on the four markers if they are all detected
    if len(marker_centers) == 4:
        frame = overlay_video_on_markerss(frame, marker_centers, ProjVid_Frame)
        cv2.aruco.drawDetectedMarkers(frame, corners_6x6, ids_6x6)

    cv2.imshow('Webcam Feed with Video Overlay and Effects', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
ProjVid_Cap.release()
cv2.destroyAllWindows()