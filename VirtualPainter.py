import cv2
import numpy as np
import os
import time
import HandTrackingModule as htm

# Brush & Eraser Thickness (set dynamically)
brushThickness = 15
eraserThickness = 50

# Load Header Images from UI folder
folderPath = "UI"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(f"Number of header images loaded: {len(overlayList)}")
header = overlayList[0]
drawColor = (255, 0, 255)

# Initialize Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize Hand Detector
detector = htm.handDetector(detectionCon=0.65, maxHands=1)
xp, yp = 0, 0

# Create Canvas for Drawing
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    img = cv2.flip(img, 1)

    # Detect hands and landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1], lmList[8][2]
        x2, y2 = lmList[12][1], lmList[12][2]
        fingers = detector.fingersUp()

        # Selection mode: Two fingers up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            if y1 < 125:
                if 350 < x1 < 440:
                    header = overlayList[0]
                    drawColor = (0, 0, 255)   # Red
                elif 530 < x1 < 650:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)   # Blue
                elif 700 < x1 < 820:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)   # Green
                elif 880 < x1 < 1000:
                    header = overlayList[3]
                    drawColor = (255, 255, 0) # Yellow
                elif 1050 < x1 < 1170:
                    header = overlayList[4]
                    drawColor = (0, 0, 0)     # Eraser (Black)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # Drawing mode: Only index finger up
        elif fingers[1] and not fingers[2]:
            # Default: medium brush
            brushThickness = 15
            # If thumb also up, thick brush
            if fingers[0]:
                brushThickness = 30
            # If pinky also up, thin brush (but not both thumb and pinky)
            elif fingers[4]:
                brushThickness = 5

            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            # Draw line on both img and canvas
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

        # Clear board with gesture: all fingers up
        if all(f == 1 for f in fingers):
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)
            print("Canvas cleared by gesture!")

    # Combine canvas and img
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)
    img[0:137, 0:1280] = header
    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Image", img)

    # Save drawing as PNG with 's' key
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break
    elif key == ord('s'):
        filename = f'Whiteboard_{int(time.time())}.png'
        cv2.imwrite(filename, imgCanvas)
        print(f"Drawing saved as {filename}")

cap.release()
cv2.destroyAllWindows()
