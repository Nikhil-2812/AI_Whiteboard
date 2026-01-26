# AI Whiteboard – Gesture-Based Drawing Application

AI Whiteboard is a computer vision based drawing application that allows users to create digital sketches using hand gestures captured through a webcam. The system uses MediaPipe for hand tracking and OpenCV for real-time image processing.

This project demonstrates the practical use of AI, computer vision, and gesture recognition in building an interactive virtual drawing tool.

# Features

- Real-time Hand Gesture Recognition using MediaPipe

- Virtual Drawing Canvas controlled without mouse or touch

- Brush Size Control

    - Index + Thumb → Thick Brush

    - Index + Little Finger → Thin Brush

- Selection Mode

    - Index + Middle Finger → Activate selection mode

    - Choose colors or eraser from the toolbar

- Eraser Tool for removing drawings

- Clear Canvas

    - Show all five fingers to reset the board

- Save Drawing

    - Press S or Ctrl + S to save the canvas as a PNG file

- On-screen gesture guide for easy use

# Technologies Used

- Python

- OpenCV – Real-time computer vision

- MediaPipe – Hand tracking and landmark detection

- NumPy – Image array processing

# Installation & Setup
1️ Clone the Repository
git clone (link of the project) 
cd AI_Whiteboard

2️  Create a Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate

3️  Install Dependencies
pip install opencv-python mediapipe numpy

4️  Run the Application
python VirtualPainter.py

# How to Use
Gesture	Action
Index Finger Up	Drawing Mode
Index + Middle Fingers	Selection Mode
Index + Thumb	Thick Brush
Index + Little Finger	Thin Brush
All Five Fingers	Clear Canvas
Press S	Save Drawing

# Project Highlights

- Touchless human-computer interaction

- Real-time gesture detection

- Practical AI + CV implementation

- Beginner-friendly but impressive portfolio project

# Output

The drawing will be saved as a PNG image in the project’s working directory.
