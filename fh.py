import cv2
import numpy as np

# Load the virtual object (e.g., a simple image or shape)
virtual_object = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.circle(virtual_object, (50, 50), 40, (0, 255, 0), -1)  # Draw a green circle

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Create a QRCodeDetector instance
detector = cv2.QRCodeDetector()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for marker detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect QR codes in the frame
    retval, points, _ = detector.detectAndDecode(gray)
    
    if retval:
        print("QR Code Detected:", retval)
        print("Points:", points)
        
        if points is not None:
            points = np.array(points, dtype=np.int32).reshape(-1, 2)
            for i in range(len(points)):
                cv2.line(frame, tuple(points[i]), tuple(points[(i + 1) % len(points)]), (0, 255, 0), 2)
            
            # Overlay virtual object on detected marker
            x, y, w, h = cv2.boundingRect(points)
            virtual_object_resized = cv2.resize(virtual_object, (w, h))
            
            # Ensure that resizing doesn't cause errors
            if virtual_object_resized.shape[0] <= h and virtual_object_resized.shape[1] <= w:
                frame[y:y+h, x:x+w] = cv2.addWeighted(frame[y:y+h, x:x+w], 0.7, virtual_object_resized, 0.3, 0)
            else:
                print("Error: Virtual object size is too large.")
    else:
        print("No QR code detected.")

    # Display the resulting frame
    cv2.imshow('AR Application', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
