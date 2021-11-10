import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import preprocessing

from tensorflow.keras.optimizers import Adam
import cv2
import mediapipe as mp
import os


def merge_edges_(vertial,horizontal,epsilon=1e-7):

    merged=tf.sqrt(vertial**2 + horizontal**2 )

    merged_max= tf.reduce_max(tf.reduce_max(merged,axis=-1),-1)
    merged_min = tf.reduce_min(tf.reduce_min(merged,axis=-1),-1)
    merge_min_max = merged_max-merged_min+ epsilon
    normalized = (merged - merged_min)/merge_min_max
    return normalized

def merge_edges(vertial,horizontal,epsilon=1e-7):


    # merged=tf.sqrt(
    #     tf.clip_by_value(vertial,clip_value_min=0,clip_value_max=255) +
    #     tf.clip_by_value(horizontal,clip_value_min=0,clip_value_max=255)
    # )
    merged=tf.clip_by_value(vertial,clip_value_min=0,clip_value_max=255) \
           +tf.clip_by_value(horizontal,clip_value_min=0,clip_value_max=255)

    merged_max= tf.reduce_max(tf.reduce_max(merged,axis=-1,keepdims=True),-2,keepdims=True)
    merged_min = tf.reduce_min(tf.reduce_min(merged,axis=-1,keepdims=True),-2,keepdims=True)
    merge_min_max = merged_max-merged_min+ epsilon
    normalized = (merged - merged_min)/merge_min_max
    return normalized

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
print(cv2.__file__)


def edge_detection_model():
    model_input = tf.keras.layers.Input(shape=(50,50,1))
    edges = tf.keras.layers.Lambda( tf.image.sobel_edges)(model_input)
    merged_edges = tf.keras.layers.Lambda(lambda x:merge_edges(x[:,:,:,0,0], x[:,:,:,0,1]))(edges)

    model = tf.keras.models.Model(inputs=model_input, outputs=merged_edges)

    model.summary()
    #
    #Compliling the model
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=['accuracy'])
    return model

def area_rectangle(x1,y1,x2,y2):
    d1 = abs(x1-x2)
    d2 = abs(y1-y2)
    return d1*d2
def crop_face_from_image(image):
    with mp_face_detection.FaceDetection( model_selection=1, min_detection_confidence=0.7) as face_detection:
        #image = cv2.imread(image_file)
        shape = image.shape

        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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


def detect_edge_from_video():
    cap = cv2.VideoCapture(0)
    model = edge_detection_model()
    face_images_list = []
    original_images_list= []
    while True:
        ret,test_img=cap.read()# captures frame and returns boolean value and captured image
        if not ret:
            continue
        #test_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        for faces_detected,success in crop_face_from_image(test_img):
            if success:
                faces_detected= cv2.cvtColor(faces_detected, cv2.COLOR_BGR2GRAY)
                test_img=cv2.resize(faces_detected,(50,50))
                img = test_img #You need to load an IMG here
                img_transform= np.expand_dims(np.expand_dims(np.array(img,dtype=np.float32), axis=0),-1)

                face_images_list.append(img_transform)
                original_images_list.append(img)
                if len(face_images_list)>=10:
                    image_batch=np.concatenate(face_images_list,axis=0)
                    edges_tf = model.predict(image_batch,steps=1)
                    result_list = np.array(edges_tf).tolist()
                    for org,edg in zip(original_images_list,result_list):
                        cv2.imshow('Original ',cv2.resize(org, (300,300)))
                        cv2.imshow('Edges-TF ', cv2.resize(np.uint8(np.reshape(edg,(50,50))*255), (300,300)))
                    face_images_list = []
                    original_images_list= []

        if cv2.waitKey(10) == ord('q'): #wait until 'q' key is pressed
            break

def detect_edges_from_images(load_directory,save_directory):
    model = edge_detection_model()
    for root, dirs, files in os.walk(load_directory):
        for file in files:
            path = os.path.join(root, file)
            image = cv2.imread(path)

            for faces_detected,success in crop_face_from_image(image):
                if success:
                    faces_detected= cv2.cvtColor(faces_detected, cv2.COLOR_BGR2GRAY)

                    face_img=cv2.resize(faces_detected,(50,50))
                    reshape_image= np.expand_dims(np.expand_dims(np.array(face_img,dtype=np.float32), axis=0),-1)

                    edges = model.predict(reshape_image)

                    face_image=cv2.resize(face_img,(300,300))
                    edges_image=cv2.resize(np.squeeze(np.expand_dims(edges,-1),0),(300,300))
                    cv2.imshow('org',face_image )
                    cv2.imshow('edges',edges_image )
                    cv2.imwrite(save_directory+'/faces/' +file, face_image)
                    cv2.imwrite(save_directory +'/edges/'+file, edges_image*255)

if __name__ == '__main__': # Test the above code
    save_directory = '../image_examples/'
    directory = '../image_examples/'
    #detect_edges_from_images(directory,save_directory)
    detect_edge_from_video()



