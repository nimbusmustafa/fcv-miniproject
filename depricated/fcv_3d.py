import cv2
import numpy as np
from stl import mesh
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# Function to initialize OpenGL
def init_gl():
    glClearColor(0.0, 0.0, 0.0, 1.0)  # Set background color to black
    glEnable(GL_DEPTH_TEST)            # Enable depth testing for 3D
    print("OpenGL initialized.")

# Function to load and render the STL 3D model
def render_3d_stl_model(stl_model, rvec, tvec):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # Set up the perspective
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1.0, 0.1, 100.0)
    
    # Apply translation and rotation based on ArUco marker detection
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    # Print the translation and rotation values for debugging
    print(f"Rendering STL at translation {tvec.flatten()} and rotation {rvec.flatten()}")

    # Translate and rotate the model based on marker's position
    glTranslatef(tvec[0][0], tvec[0][1], -tvec[0][2] - 5.0)  # Adjust z-axis to move it into view
    glRotatef(np.degrees(rvec[0][0]), 1.0, 0.0, 0.0)
    glRotatef(np.degrees(rvec[0][1]), 0.0, 1.0, 0.0)
    glRotatef(np.degrees(rvec[0][2]), 0.0, 0.0, 1.0)
    
    # Scale the model down if it's too large
    glScalef(0.1, 0.1, 0.1)
    
    # Render the STL model
    glBegin(GL_TRIANGLES)
    for facet in stl_model.vectors:
        for vertex in facet:
            glVertex3fv(vertex)
    glEnd()

    # Swap buffers to display the rendered image
    glutSwapBuffers()

# Load your STL model (replace with the correct path to your .stl file)
stl_model = mesh.Mesh.from_file('cube.stl')  # Replace this with the path to your .stl file

# Print the number of triangles for debugging
print(f"STL model has {len(stl_model.vectors)} triangles.")

# Initialize OpenGL
glutInit()
glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)

# Create an OpenGL window and set its size
glutInitWindowSize(800, 600)
glutCreateWindow("3D STL Model Rendering")  # Create the window with a title

init_gl()

# Initialize OpenCV ArUco dictionary and detector parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# Create an ArUco detector object (new approach for OpenCV 4.10.0)
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Camera calibration parameters (intrinsic matrix and distortion coefficients)
# Replace these with your calibrated values
camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=float)
dist_coeffs = np.zeros((5, 1))  # Assuming no lens distortion

# Marker size in meters
marker_size = 0.05  # 5 cm

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Display function for OpenGL
def display():
    # Capture a frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        return

    # Print frame shape to verify it's being captured
    print("Frame shape:", frame.shape)
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect ArUco markers in the image using the ArUco detector object
    corners, ids, rejected = aruco_detector.detectMarkers(gray)

    if ids is not None:
        # Estimate the pose of the ArUco marker using the cv2.aruco method
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
        
        for rvec, tvec in zip(rvecs, tvecs):
            # Draw the detected marker
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Render the 3D STL model above the detected ArUco marker
            render_3d_stl_model(stl_model, rvec, tvec)
            
            # Print when the model is being rendered on the webcam feed
            print("3D model is being rendered on the webcam feed.")
    
    # Display the webcam feed in a separate window
    cv2.imshow('Webcam Feed', frame)
    
    # Check if the webcam frame is displayed
    if frame is not None:
        print("Webcam feed is being displayed.")

    # Force OpenGL to refresh and redraw
    glutPostRedisplay()

# Set the display function for OpenGL
glutDisplayFunc(display)

# Start the OpenGL main loop
glutMainLoop()

# Release the webcam and close OpenCV windows after exit
cap.release()
cv2.destroyAllWindows()
