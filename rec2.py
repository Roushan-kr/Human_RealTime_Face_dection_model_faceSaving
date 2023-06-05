import cv2
import dlib
import numpy as np
from imutils.video import FPS
from imutils.video import WebcamVideoStream

# Load the pre-trained models for face recognition and object detection
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

# Load the classes for object detection
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load the video stream
vs = WebcamVideoStream(src=1).start() # 1 for webcam
fps = FPS().start()

# Initialize variables to track detected objects and recognized faces
prev_frame_faces = []
prev_frame_objects = []

# Loop over frames from the video stream
while True:
    frame = vs.read()
    if frame is None:
        break

    # Resize the frame and convert it to a blob
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
            scores = detection[5]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter detections by confidence threshold and class label
            if confidence > 0.5 and classes[class_id] == 'person':
                center_x, center_y, width, height = detection[:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                object_locations.append((x, y, int(width), int(height)))
                object_classes.append(classes[class_id])

    # Detect faces in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # Initialize lists for face recognition results
    face_descriptors = []
    face_locations = []

    # Iterate over the detected faces
    for face in faces:
        landmarks = predictor(gray, face)
        face_descriptor = face_recognizer.compute_face_descriptor(frame, landmarks, num_jitters=1)
        face_descriptors.append(face_descriptor)
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        face_locations.append((y, x + w, y + h, x))

    # Match and update the object and face lists
    matched_faces = []
    matched_objects = []

    for face_loc in face_locations:
        for obj_loc in object_locations:
            if obj_loc[0] < face_loc[0] < obj_loc[0] + obj_loc[2] and obj_loc[1] < face_loc[1] < obj_loc[1] + obj_loc[3]:
                matched_faces.append(face_loc)
                matched_objects.append(obj_loc)
                break

    prev_frame_faces = matched_faces
    prev_frame_objects = matched_objects

    # Draw bounding boxes and labels for object detection
    for (x, y, w, h) in object_locations:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw bounding boxes and labels for face recognition
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(frame, "Face", (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow("Real-time Face Recognition and Object Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    # Break the loop on 'q' key press
    if key == ord("q"):
        break

    # Update the FPS counter
    fps.update()

# Stop the video stream and display the final FPS information
fps.stop()
vs.stop()

# Close all windows
cv2.destroyAllWindows()
