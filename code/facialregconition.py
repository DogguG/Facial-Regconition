import cv2
from fer import FER

# Load Haar Cascade for face detection
haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# Initialize emotion detector
emotion_detector = FER(mtcnn=True)

# Capture video feed from the camera
cam = cv2.VideoCapture(0)

while True:
    _, img = cam.read()
    text = "Face not detected"
    # Convert frame from BGR to Grayscale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces using Haar Cascade
    faces = haar_cascade.detectMultiScale(grayImg, 1.3, 4)
    
    # Detect emotions if face is found
    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        emotion, score = emotion_detector.top_emotion(face_img)  # Get the top emotion
        text = f"Emotion: {emotion}" if emotion else "Face detected"
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Display emotion on the image
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Display the output window and press ESC to exit
    cv2.imshow("Emotion Detection", img)
    key = cv2.waitKey(10)
    if key == 27:  # ESC key to exit
        break

cam.release()
cv2.destroyAllWindows()
