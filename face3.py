import cv2
import numpy as np
from keras.models import load_model

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # use path add for this 

cap = cv2.VideoCapture(1) # 0 for main cam or 1 for external
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX

model = load_model('keras_model.h5') # here you have to generate your self learnedd h5 forment ml file or use teachable machine to do it 

class_names = ["person1", "person2", "person3"]

recognized_faces = []
frame_counter = 0

while True:
    success, imgOriginal = cap.read()
    if not success:
        break

    frame_counter += 1

    if frame_counter > 3:
        recognized_faces = []
        frame_counter = 1

    faces = facedetect.detectMultiScale(imgOriginal, 1.8, 2)

    new_faces = []
    for x, y, w, h in faces:
        crop_img = imgOriginal[y:y + h, x:x + w]
        img = cv2.resize(crop_img, (224, 224))
        img = img / 255.0  # Normalize the image

        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        probability_value = np.amax(prediction)

        if probability_value > 0.6:  # Adjust the threshold as needed
            is_duplicate = False
            for face in recognized_faces:
                if (
                    x >= face['box'][0] - face['box'][2] * 0.2
                    and y >= face['box'][1] - face['box'][3] * 0.2
                    and (x + w) <= face['box'][0] + face['box'][2] * 1.2
                    and (y + h) <= face['box'][1] + face['box'][3] * 1.2
                ):
                    is_duplicate = True
                    break
            if not is_duplicate:
                new_faces.append({
                    'class_index': class_index,
                    'probability': probability_value,
                    'box': (x, y, w, h)
                })

    recognized_faces.extend(new_faces)

    for face in recognized_faces:
        class_index = face['class_index']
        probability_value = face['probability']
        x, y, w, h = face['box']

        if class_index < len(class_names):
            name = class_names[class_index]
        else:
            name = "person" + str(class_index)

        cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(imgOriginal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
        cv2.putText(imgOriginal, name, (x, y - 10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(imgOriginal, str(round(probability_value * 100, 2)) + "%", (180, 75), font, 0.75, (255, 0, 0),
                    2, cv2.LINE_AA)

    cv2.imshow("Result", imgOriginal)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
