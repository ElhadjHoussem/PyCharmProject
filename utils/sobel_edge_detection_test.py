import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import cv2
import mediapipe as mp
import os
from sklearn.preprocessing import normalize

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
print(cv2.__file__)


model_input = tf.keras.layers.Input(shape=(50,50,1))
out_put = tf.keras.layers.Lambda( tf.image.sobel_edges)(model_input)
#out_put = tf.image.sobel_edges(model_input)

model = tf.keras.models.Model(inputs=model_input, outputs=out_put)
model.summary()
#
#Compliling the model
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])
def crop_face_from_image(image):
    with mp_face_detection.FaceDetection( model_selection=1, min_detection_confidence=0.7) as face_detection:
        #image = cv2.imread(image_file)
        shape = image.shape

        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.detections:
            for detection in results.detections:
                offset_x =0.05
                offset_y =0.1
                bbbox = detection.location_data.relative_bounding_box

                # (x,y) upper  left corner
                bbbox_x_min = 0.01 if bbbox.xmin<0  else bbbox.xmin-offset_x if  bbbox.xmin-offset_x>0 else bbbox.xmin
                bbbox_y_min = 0.01 if bbbox.ymin <0  else bbbox.ymin-offset_y if  bbbox.ymin-offset_y>0 else bbbox.ymin

                # (x,y) buttom right left corner
                bbbox_x_max =  1 if bbbox.xmin >1 else bbbox.xmin+offset_x + bbbox.width if bbbox.xmin+offset_x+ bbbox.width <1 else bbbox.xmin + bbbox.width
                bbbox_y_max =  1 if bbbox.ymin >1 else  bbbox.ymin + bbbox.height

                rect_start_point =mp_drawing._normalized_to_pixel_coordinates(
                  bbbox_x_min,
                  bbbox_y_min,
                  shape[1],
                  shape[0])
                if rect_start_point is None: rect_start_point = 0,0
                rect_end_point = mp_drawing._normalized_to_pixel_coordinates(
                   bbbox_x_max,
                   bbbox_y_max,
                   shape[1], shape[0])
                if rect_end_point is None: rect_end_point = 0,0

                def area_rectangle(x1,y1,x2,y2):
                    d1 = abs(x1-x2)
                    d2 = abs(y1-y2)
                    return d1*d2
                if area_rectangle(rect_start_point[1],rect_end_point[1],rect_start_point[0],rect_end_point[0])>1000:
                    crop_img = image[
                       rect_start_point[1]:rect_end_point[1],
                       rect_start_point[0]:rect_end_point[0]
                       ]
                    yield crop_img,True
                else:
                    yield None, False
        else:
            print('no Face Detected')
            yield None, False

if __name__ == '__main__': # Test the above code
    #face_haar_cascade = cv2.CascadeClassifier('../downloads/haarcascade/haarcascade_frontalface_alt2.xml')
    cap = cv2.VideoCapture(0)
    while True:
        ret,test_img=cap.read()# captures frame and returns boolean value and captured image

        if not ret:
            continue
        test_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        #test_img=cv2.resize(test_img,(50,50))

        #faces_detected = face_haar_cascade.detectMultiScale(test_img, 1.32, 5)
        for faces_detected,success in crop_face_from_image(test_img):
            if success:

                test_img=cv2.resize(faces_detected,(50,50))

                img = test_img #You need to load an IMG here
                edges_opencv = cv2.Sobel(np.uint8(img), ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)


                img_transform= tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(img,dtype=tf.float32), axis=0),-1)

                edges_tf = model.predict(img_transform,steps=1)

                edges_tf_x = edges_tf[0,:,:,:,0]
                edges_tf_y = edges_tf[0,:,:,:,1]

                edges_xy = np.sqrt(edges_tf_x**2+edges_tf_y**2)
                shape = edges_xy.shape
                fusion = np.sqrt(edges_tf_x**2+edges_tf_y**2)
                norm_fusion = np.linalg.norm(fusion)
                normalized_fusion = tf.keras.utils.normalize(fusion,axis=0,order=2)
                edges_tf_xy = np.reshape(normalized_fusion,newshape=shape)

                combine = np.expand_dims(img,-1) + edges_tf_xy
                shape = combine.shape
                norm_enhanced = np.linalg.norm(combine)
                img_enhanced = np.reshape(combine/2,newshape=shape)

                #img_enhanced = cv2.addWeighted(img,0.7,edges_tf_xy,0.5,addweight)

                edges_tf_x = edges_tf_x/np.linalg.norm(edges_tf_x)
                edges_tf_y = edges_tf_y/np.linalg.norm(edges_tf_y)
                cv2.imshow('Original ',cv2.resize(img, (300,300)))
                cv2.imshow('Edges-OpenCV.sobel ', cv2.resize(edges_opencv, (300,300)))
                cv2.imshow('Edges-TF_x ', cv2.resize(edges_tf_x, (300,300)))
                cv2.imshow('Edges-TF_y ', cv2.resize(edges_tf_y, (300,300)))
                cv2.imshow('Edges-TF_xy ', cv2.resize(edges_tf_xy, (300,300)))
                cv2.imshow('Original + Edges-TF_Canny ', cv2.resize(img_enhanced, (300,300)))

        if cv2.waitKey(10) == ord('q'): #wait until 'q' key is pressed
            break
