## Prerequisites
* Aruco Marker printouts of dictionary 4x4, 5x5 and 6x6. Tags should be numbered 1 to 4. Available in the PDF given

## Presentation link
https://www.canva.com/design/DAGV5F9Lei4/a6VFa9CnG26kokhFInq-ag/edit


# 🎥 ArUco Marker Interactive Video & 3D Projection

This project uses **OpenCV ArUco markers** to control interactive video overlays and 3D model rendering in real-time.

It has **two modes of operation**:

1. **`final2.py`** → Augmented video playback with marker-based controls.
2. **`gui.py`** → Streamlit GUI for switching between video overlay mode and 3D model projection mode.

---

## ✨ Features

* 📌 **Marker-based Controls (final2.py)**

  * Switch between multiple videos using markers (up/down arrows).
  * Change video effects (grayscale, blur, canny, negative).
  * Overlay video frames on a detected 4-marker square.

* 🖥️ **Streamlit GUI (gui.py)**

  * Toggle between **Video Overlay Mode** and **3D Model Projection Mode**.
  * Load custom `.mp4` videos or `.stl` 3D models.
  * Adjust scale factor of 3D model dynamically.

* 🎯 **Real-Time Augmented Reality** using OpenCV, ArUco, and Perspective Warping.

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/aruco-video-3d.git
cd aruco-video-3d
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
opencv-python
opencv-contrib-python
numpy
streamlit
numpy-stl
```

---

## ▶️ Usage

### **1. Run Marker-Controlled Video Overlay**

```bash
python final2.py
```

* Show **4x4 ArUco markers (ID 0, 1, 2)** to overlay arrow controls:

  * **ID 0** → Up arrow → Next video
  * **ID 1** → Down arrow → Previous video
  * **ID 2** → Edit arrow → Change video effect
* Show **6x6 ArUco markers (IDs 0, 1, 2, 3)** in a square → Projects current video inside the marker frame.

---

### **2. Run Streamlit GUI**

```bash
streamlit run gui.py
```

* Sidebar → Select mode:

  * **Video Overlay** → Pick a `.mp4` file, overlay on marker.
  * **3D Model** → Pick an `.stl` file, adjust scale, project on marker.

---

## 📂 Repository Structure

```
.
├── final2.py        # Marker-based video control + overlay
├── gui.py           # Streamlit GUI for video/model overlay
├── requirements.txt # Python dependencies
├── assets/          # (Optional) arrow images, markers, videos, stl files
```

---

## ⚙️ How It Works

### 🔹 `final2.py` (Marker-Based Video Player)

1. Webcam captures frames.
2. Detects **4x4 ArUco markers (0,1,2)** → overlays arrow icons.
3. If a marker disappears for 2s → action triggered (switch video / change effect).
4. Detects **6x6 ArUco markers (0,1,2,3)** → overlays video on the marker plane using perspective warp.

### 🔹 `gui.py` (Streamlit GUI)

1. User selects mode:

   * **Video Overlay** → Projects video on detected marker.
   * **3D Model** → Loads `.stl` mesh, applies scale, projects onto marker using OpenCV’s `projectPoints`.
2. Streamlit displays real-time processed frames.


## 📷 Demo

* 🎬 **Video Overlay Example**
  ![alt text](<Images/Screenshot 2024-11-08 195918.png>)




---
* 🧩 **3D Model Example**

  
  ![alt text](Images/3drender.jpeg)
---

## 🚀 Future Improvements

* Add support for multiple simultaneous videos/models.
* Use **TensorFlow Lite or MediaPipe** for gesture-based control instead of only marker-based.
* Save user sessions in Streamlit for persistent settings.

---



