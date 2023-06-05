import cv2
import dlib
import numpy as np
from imutils.video import FPS
from imutils.video import WebcamVideoStream
import os

# Load the pre-trained models for face recognition and object detection
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat") #  here you need it to download from internet 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #  here you need it to download from internet 
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights") # here you need it to download from internet 

# Load the classes for object detection
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Taking username and try to make it capture
nameID = str(input("Enter Your Name: ")).lower()
path = 'images/' + nameID
isExist = os.path.exists(path) # it ceate new file of that uName
if isExist:
    print("Name Already Taken")
    nameID = str(input("Enter Your Name Again: "))
else:
    os.makedirs(path)

# Load the video stream
vs = WebcamVideoStream(src=0).start()  # my cam source
fps = FPS().start()
count = 0

# Loop over frames from the video stream
while True:
    frame = vs.read()
    if frame is None:
        break

    # Resize the frame and convert it to a blob
    frame = cv2.resize(frame, (416, 416))
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Pass the blob through the network to perform object detection
    net.setInput(blob)
    layer_outputs = net.forward(net.getUnconnectedOutLayersNames())

    # Initialize lists for object detection results
    object_locations = []
    object_classes = []

    # Iterate over the output layers and detect objects
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter detections by confidence threshold and class label
            if confidence > 0.5 and classes[class_id] == 'person':
                center_x, center_y, width, height = detection[:4] * np.array([416, 416, 416, 416])
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                object_locations.append((x, y, int(width), int(height)))
                object_classes.append(classes[class_id])

    # Detect faces in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # Iterate over the detected faces
    for face in faces:
        landmarks = predictor(gray, face)
        face_descriptor = face_recognizer.compute_face_descriptor(frame, landmarks, num_jitters=10)
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())

        count += 1
        name = f'{path}/{nameID}_{count}.jpg'
        print("Creating Image: " + name)
        cv2.imwrite(name, frame[y:y + h, x:x + w])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if count >= 100:
            break

    # Draw bounding boxes and labels for object detection
    for (x, y, w, h) in object_locations:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Person", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Real-time Face Recognition and Object Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    # Break the loop on 'q' key press or when 500 face images are saved
    if key == ord("q") or count >= 150:
        break

    # Update the FPS counter
    fps.update()

# Stop the video stream and display the final FPS information
fps.stop()
vs.stop()

# Close all windows
cv2.destroyAllWindows()
