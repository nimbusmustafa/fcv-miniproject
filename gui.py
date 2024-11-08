import streamlit as st
import cv2
import numpy as np
from stl import mesh
import os

# Function to project and draw a 3D model on the ArUco marker
def render_3d_model_on_marker(frame, stl_model, rvec, tvec, camera_matrix, dist_coeffs, scale_factor=0.01):
    object_points = np.array([vertex for facet in stl_model.vectors for vertex in facet], dtype=np.float32) * scale_factor
    
    # Project the 3D points to 2D image points
    imgpts, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = np.int32(imgpts).reshape(-1, 2)
    
    # Draw the projected points as polygons
    for i in range(0, len(imgpts), 3):
        cv2.polylines(frame, [imgpts[i:i+3]], isClosed=True, color=(0, 255, 0), thickness=2)

# Function to overlay video on ArUco marker
def FindMarkerCoordinates(Frame):
    GrayFrame = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    ArucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    Parameters = cv2.aruco.DetectorParameters()
    Corners, IDs, _ = cv2.aruco.detectMarkers(GrayFrame, ArucoDict, parameters=Parameters)
    if IDs is None or len(IDs) == 0:
        return None
    return Corners[0][0]

def ProjectiveTransform(Frame, Coordinates, TransFrameShape):
    Height, Width = Frame.shape[:2]
    InitialPoints = np.float32([[0, 0], [Width-1, 0], [0, Height-1], [Width-1, Height-1]])
    FinalPoints = np.float32([Coordinates[0], Coordinates[1], Coordinates[3], Coordinates[2]])
    ProjectiveMatrix = cv2.getPerspectiveTransform(InitialPoints, FinalPoints)
    TransformedFrame = cv2.warpPerspective(Frame, ProjectiveMatrix, TransFrameShape[::-1])
    return TransformedFrame

def OverlapFrames(BaseFrame, SecFrame, MarkerCoordinates):
    TransformedFrame = ProjectiveTransform(SecFrame, MarkerCoordinates, BaseFrame.shape[:2])
    SecFrame_Mask = np.zeros(BaseFrame.shape, dtype=np.uint8)
    cv2.fillConvexPoly(SecFrame_Mask, np.asarray(MarkerCoordinates, dtype=np.int32), (255, )*BaseFrame.shape[2])
    BaseFrame = cv2.bitwise_and(BaseFrame, cv2.bitwise_not(SecFrame_Mask))
    OverlapedFrame = cv2.bitwise_or(BaseFrame, TransformedFrame)
    return OverlapedFrame

# Main Streamlit application
def main():
    st.title("Aruco Marker Application")

    # Button for selecting the mode (Video or Model)
    option = st.sidebar.selectbox(
        "Select Mode",
        ("None", "Video Overlay", "3D Model")
    )

    if option == "None":
        st.write("Please select a mode from the sidebar to continue.")
        return

    # Initialize the webcam only once and reuse it
    cap = st.session_state.get('cap', None)
    
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        st.session_state['cap'] = cap

    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    # Dropdown to select video or model after selecting the mode
    if option == "3D Model":
        # Get a list of STL files in the current directory (or a specific folder)
        stl_files = [file for file in os.listdir() if file.endswith('.stl')]
        selected_stl = st.sidebar.selectbox("Select 3D Model", stl_files)
        
        # Slider for adjusting the scale factor with a finer step (0.00001)
        scale_factor = st.sidebar.slider("Select Scale for 3D Model", min_value=0.00005, max_value=0.005, value=0.0005, step=0.00001, format="%.5f")
        
        st.write(f"3D Model Mode: {selected_stl} loaded with scale factor {scale_factor:.5f}.")
        stl_model = mesh.Mesh.from_file(selected_stl)

    if option == "Video Overlay":
        # Get a list of video files in the current directory (or a specific folder)
        video_files = [file for file in os.listdir() if file.endswith('.mp4') or file.endswith('.avi')]
        selected_video = st.sidebar.selectbox("Select Video", video_files)
        st.write(f"Video Overlay Mode: {selected_video} selected.")
        ProjVid_Cap = cv2.VideoCapture(selected_video)
        if not ProjVid_Cap.isOpened():
            st.error("Error: Could not load the video file.")
            return

    # Start video streaming
    FRAME_WINDOW = st.image([])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        parameters = cv2.aruco.DetectorParameters()
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            if option == "3D Model":
                # Estimate pose of the ArUco marker
                camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=float)
                dist_coeffs = np.zeros((5, 1))  # Assuming no lens distortion
                marker_size = 0.01  # 1 cm marker size
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
                for rvec, tvec in zip(rvecs, tvecs):
                    # Render 3D model on the marker with the selected scale factor
                    render_3d_model_on_marker(frame, stl_model, rvec, tvec, camera_matrix, dist_coeffs, scale_factor)

            elif option == "Video Overlay":
                # Read projection video frame
                retProjVid, ProjVid_Frame = ProjVid_Cap.read()
                if not retProjVid:
                    ProjVid_Cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    retProjVid, ProjVid_Frame = ProjVid_Cap.read()
                MarkerCoordinates = FindMarkerCoordinates(frame)
                if MarkerCoordinates is not None:
                    frame = OverlapFrames(frame, ProjVid_Frame, MarkerCoordinates)

            # Draw detected markers on the frame
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Show the frame in the Streamlit app
        FRAME_WINDOW.image(frame, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video resources
    cap.release()
    if option == "Video Overlay":
        ProjVid_Cap.release()

if __name__ == "__main__":
    main()
