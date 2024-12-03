import cv2
import numpy as np
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)  # Detect a single hand
mp_drawing = mp.solutions.drawing_utils  # Utility to draw landmarks

# Global variables for hand interaction and timing
detected_marker_corners = None
finger_on_marker = False
finger_on_marker_start_time = None
button_clicked = False
required_time_on_marker = 1  # Set the required time to 1.5 seconds

# Load the up and down arrow images
up_arrow_img = cv2.imread('arrowup.png', cv2.IMREAD_UNCHANGED)
down_arrow_img = cv2.imread('arrowdown.png', cv2.IMREAD_UNCHANGED)

# Load the videos
video_files = ['video1.mp4', 'video2.mp4', 'output.mp4', 'rohit.mp4']  # Add more video paths here
current_video_index = 0
ProjVid_Cap = cv2.VideoCapture(video_files[current_video_index])

# Function to switch the video
def switch_video(direction):
    global current_video_index, ProjVid_Cap
    current_video_index = (current_video_index + direction) % len(video_files)
    ProjVid_Cap = cv2.VideoCapture(video_files[current_video_index])
    print(f"Switched to video {current_video_index + 1}")

# Function to rotate video clockwise
def rotate_video_clockwise(video_frame):
    return cv2.rotate(video_frame, cv2.ROTATE_90_CLOCKWISE)

# Function to overlay the video at the center of the ArUco marker
def overlay_video_on_marker(frame, marker_centers, video_frame):
    if len(marker_centers) != 4:
        return frame  # If not exactly four centers, return the original frame

    h, w = video_frame.shape[:2]
    pts_src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")

    # Assign the corners based on relative screen position
    pts_dst = np.array(marker_centers, dtype="float32")

    # Get the perspective transform matrix and warp the video onto the ArUco marker area
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped_video = cv2.warpPerspective(video_frame, M, (frame.shape[1], frame.shape[0]))

    # Create a mask for the video and blend it into the frame
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(pts_dst), (255, 255, 255))
    frame_bg = cv2.bitwise_and(frame, cv2.bitwise_not(mask))
    frame_fg = cv2.bitwise_and(warped_video, mask)
    frame = cv2.add(frame_bg, frame_fg)

    return frame

# Function to check if a point (x, y) is inside a polygon (marker)
def is_finger_over_marker(finger_tip, marker_corners):
    if marker_corners is not None:
        marker_polygon = np.array(marker_corners, dtype=np.int32).reshape((4, 2))
        return cv2.pointPolygonTest(marker_polygon, finger_tip, False) >= 0
    return False

# Function to overlay the arrow images on the ArUco markers
def overlay_arrow_on_marker(frame, corners, arrow_img):
    h, w = arrow_img.shape[:2]
    pts_src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")

    # Warp the arrow image onto the marker
    M = cv2.getPerspectiveTransform(pts_src, corners)
    warped_arrow = cv2.warpPerspective(arrow_img, M, (frame.shape[1], frame.shape[0]))

    # Create a mask for the arrow image and overlay it
    arrow_mask = warped_arrow[:, :, 3]  # Alpha channel for transparency
    arrow_img_rgb = warped_arrow[:, :, :3]  # RGB channels
    frame[arrow_mask > 0] = arrow_img_rgb[arrow_mask > 0]

    return frame

# Main program
if __name__ == "__main__":
    # Initialize OpenCV ArUco dictionary and detector parameters
    aruco_dict_6x6_50 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    aruco_dict_4x4_50 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Read the current frame from the video
        retProjVid, ProjVid_Frame = ProjVid_Cap.read()
        if not retProjVid:
            ProjVid_Cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video if it ends
            retProjVid, ProjVid_Frame = ProjVid_Cap.read()

        # Rotate the video frame clockwise
        ProjVid_Frame = rotate_video_clockwise(ProjVid_Frame)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners_6x6, ids_6x6, _ = cv2.aruco.detectMarkers(gray, aruco_dict_6x6_50, parameters=parameters)
        corners_4x4, ids_4x4, _ = cv2.aruco.detectMarkers(gray, aruco_dict_4x4_50, parameters=parameters)

        # Detect hand landmarks using MediaPipe Hands
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        # Check if 4 ArUco markers are detected for video overlay
        if ids_6x6 is not None and len(corners_6x6) >= 4:
            marker_centers = [np.mean(corner[0], axis=0) for corner in corners_6x6]
            frame = overlay_video_on_marker(frame, marker_centers, ProjVid_Frame)
            cv2.aruco.drawDetectedMarkers(frame, corners_6x6, ids_6x6)

        # Handle the up and down buttons (4x4 ArUco markers)
        if ids_4x4 is not None:
            for marker_corners, marker_id in zip(corners_4x4, ids_4x4):
                if marker_id == 0:  # Up arrow marker
                    frame = overlay_arrow_on_marker(frame, marker_corners[0], up_arrow_img)
                    detected_marker_corners = marker_corners[0]  # Save corners for interaction
                elif marker_id == 1:  # Down arrow marker
                    frame = overlay_arrow_on_marker(frame, marker_corners[0], down_arrow_img)
                    detected_marker_corners = marker_corners[0]  # Save corners for interaction

        # Handle hand gesture interactions
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                index_finger_tip = hand_landmarks.landmark[8]
                h, w, _ = frame.shape
                finger_tip_pixel = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))

                # Check if the finger is over the detected marker
                if is_finger_over_marker(finger_tip_pixel, detected_marker_corners):
                    if not finger_on_marker and not button_clicked:
                        finger_on_marker = True
                        finger_on_marker_start_time = time.time()
                    elif finger_on_marker and not button_clicked:
                        elapsed_time = time.time() - finger_on_marker_start_time
                        if elapsed_time >= required_time_on_marker:
                            # Detect which marker is clicked and switch videos accordingly
                            if marker_id == 0:
                                print("Next video!")
                                switch_video(1)
                            elif marker_id == 1:
                                print("Previous video!")
                                switch_video(-1)
                            button_clicked = True
                else:
                    finger_on_marker = False
                    button_clicked = False

        # Display the webcam feed
        cv2.imshow('Aruco Video Overlay', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    ProjVid_Cap.release()
    cv2.destroyAllWindows()
