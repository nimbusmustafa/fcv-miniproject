import cv2
import numpy as np

def FindMarkerCoordinates(Frame):
    # Detecting marker
    GrayFrame = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    ArucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    Parameters = cv2.aruco.DetectorParameters()
    Corners, IDs, RejectedImgPoints = cv2.aruco.detectMarkers(GrayFrame, ArucoDict, parameters=Parameters)

    # If no ArUco marker found, return None
    if IDs is None or len(IDs) == 0:
        return None

    # Returning coordinates of the first detected marker
    return Corners[0][0]

def ProjectiveTransform(Frame, Coordinates, TransFrameShape):
    Height, Width = Frame.shape[:2]
    InitialPoints = np.float32([[0, 0], [Width-1, 0], [0, Height-1], [Width-1, Height-1]])
    FinalPoints = np.float32([Coordinates[0], Coordinates[1], Coordinates[3], Coordinates[2]])
    
    ProjectiveMatrix = cv2.getPerspectiveTransform(InitialPoints, FinalPoints)
    TransformedFrame = cv2.warpPerspective(Frame, ProjectiveMatrix, TransFrameShape[::-1])
    
    return TransformedFrame

def OverlapFrames(BaseFrame, SecFrame, MarkerCoordinates):
    # Finding transformed image
    TransformedFrame = ProjectiveTransform(SecFrame, MarkerCoordinates, BaseFrame.shape[:2])

    # Overlapping frames
    SecFrame_Mask = np.zeros(BaseFrame.shape, dtype=np.uint8)
    cv2.fillConvexPoly(SecFrame_Mask, np.asarray(MarkerCoordinates, dtype=np.int32), (255, )*BaseFrame.shape[2])

    BaseFrame = cv2.bitwise_and(BaseFrame, cv2.bitwise_not(SecFrame_Mask))
    OverlapedFrame = cv2.bitwise_or(BaseFrame, TransformedFrame)
    
    return OverlapedFrame

if __name__ == "__main__":
    # Capturing video from webcam
    Webcam_Cap = cv2.VideoCapture(0)

    # Reading video for projection from directory
    ProjVid_Cap = cv2.VideoCapture("rohit.mp4")

    SkippedFrames = []  # record of skipped frames will be kept here

    while True:
        # Checking if webcam and video are opened.
        if not Webcam_Cap.isOpened():
            print("Not able to access webcam.")
            break
        if not ProjVid_Cap.isOpened():
            print("Not able to read projection video.")
            break

        # Reading frames
        retWebcam, Webcam_Frame = Webcam_Cap.read()
        retProjVid, ProjVid_Frame = ProjVid_Cap.read()

        # If webcam frame is not read correctly, break loop
        if not retWebcam:
            break

        # Restart the projection video if finished
        if not retProjVid:
            ProjVid_Cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            retProjVid, ProjVid_Frame = ProjVid_Cap.read()

        # Detecting ArUco marker in the frame
        MarkerCoordinates = FindMarkerCoordinates(Webcam_Frame)

        if MarkerCoordinates is not None:
            # Overlay the video only if the marker is detected
            Webcam_Frame = OverlapFrames(Webcam_Frame, ProjVid_Frame, MarkerCoordinates)

        # Displaying Output video
        cv2.imshow("Augmented Reality Overlay", Webcam_Frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Releasing video objects and destroying windows
    Webcam_Cap.release()
    ProjVid_Cap.release()
    cv2.destroyAllWindows()