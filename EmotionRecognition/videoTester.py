import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing import image

PATH ="../SavedModels/AffectNet/AffectNet64x64_3/AffNet_01.h5"

#load model
# model = tf.keras.models.model_from_json(open(PATH+".json", "r").read(),
#                                         custom_objects={
#                                             "GlorotUniform": tf.keras.initializers.glorot_uniform,
#                                             "BatchNormalizationV1":tf.keras.layers.BatchNormalization
#                                         })
#load weights
# model.load_weights(PATH+'.h5')

model = tf.keras.models.load_model(PATH)
face_haar_cascade = cv2.CascadeClassifier('../downloads/haarcascade/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
emotions = ['Neutral','Happy','Sad','Surprise','Fear','Disgust','Anger','Contempt']

emotion_list= ",".join(["{:7s}:{:1.2f}".format(emopred[0],emopred[1]) for emopred in zip(emotions,[0 for _ in range(11)])])

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img_pixel ',gray_img)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)


    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray=gray_img[y:y+w,x:x+h]  #cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(64,64))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        #img_pixels /= 255

        predictions = model.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])

        #emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
        emotion_list= ",".join(["{:7s}:{:1.2f}".format(emopred[0],emopred[1]) for emopred in zip(emotions,predictions[0])])
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


    resized_img = cv2.resize(test_img, (1000, 700))
    y_start = int(100)
    for emotions_string in emotion_list.split(','):
        cv2.putText(resized_img, emotions_string, (int(20), y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        y_start+=20
    cv2.imshow('Facial emotion analysis ',resized_img)

    if cv2.waitKey(10) == ord('q'): #wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows()
