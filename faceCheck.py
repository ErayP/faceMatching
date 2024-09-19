import cv2
import numpy as np
from keras.models import model_from_json
from PIL import Image


model = model_from_json(open("model.json", "r").read())
model.load_weights("faces.weights.h5")


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


width, height = 223, 223


cap = cv2.VideoCapture(0)


while True:

    success, img = cap.read()
    if not success:
        break

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


        face_img = img_gray[y:y + h, x:x + w]
        face_img = np.array(Image.fromarray(face_img).resize((width, height)))
        face_img = face_img / 255.0
        face_img = face_img.reshape(1, width, height, 1)


        prediction = model.predict(face_img)
        predicted_label = np.argmax(prediction)
        match_probability = np.max(prediction) * 100


        if predicted_label == 0:
            text = f"face mesh ({match_probability:.2f}%)"
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            text = f"Face not mesh ({match_probability:.2f}%)"
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    cv2.imshow("Yuz Tanima", img)


    if cv2.waitKey(20) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
