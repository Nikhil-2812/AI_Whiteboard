import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    ret, frame = cap.read()
    if ret:
        print(f"Camera index {i} works.")
        cv2.imshow("Camera Test", frame)
        cv2.waitKey(3000)
        cap.release()
        cv2.destroyAllWindows()
        break
    cap.release()
else:
    print("No working camera found.")
