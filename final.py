import cv2
import numpy as np
from keras.models import load_model

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX

model = load_model('keras_model.h5')

class_names = ["member", "Prof", "other"]

while True:
    success, imgOriginal = cap.read()
    if not success:
        break

    faces = facedetect.detectMultiScale(imgOriginal, 1.8, 2)

    for x, y, w, h in faces:
        crop_img = imgOriginal[y:y + h, x:x + w]
        img = cv2.resize(crop_img, (224, 224))
        img = img / 255.0  # Normalize the image

        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        classIndex = np.argmax(prediction)
        probabilityValue = np.amax(prediction)

        cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(imgOriginal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
        cv2.putText(imgOriginal, class_names[classIndex], (x, y - 10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(imgOriginal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (255, 0, 0), 2,
                    cv2.LINE_AA)

    cv2.imshow("Result", imgOriginal)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
