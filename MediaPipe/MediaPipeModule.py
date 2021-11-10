import cv2
import mediapipe as mp
import os
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def selfiSegmentation():
    pass
def poseEstimation():
    pass
def detect_face_from_image(image):
    with mp_face_detection.FaceDetection( model_selection=1, min_detection_confidence=0.5) as face_detection:
        shape = image.shape
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.detections:
            for detection in results.detections:
                offset_x =0.05
                offset_y =0.1
                bbbox = detection.location_data.relative_bounding_box

                # (x,y) upper  left corner
                bbbox_x_min = bbbox.xmin-offset_x if  bbbox.xmin-offset_x>0 else bbbox.xmin
                bbbox_y_min = bbbox.ymin-offset_y if  bbbox.ymin-offset_y>0 else bbbox.ymin

                # (x,y) buttom right left corner
                bbbox_x_max = bbbox.xmin+offset_x + bbbox.width if bbbox.xmin+offset_x+ bbbox.width <1 else bbbox.xmin + bbbox.width
                bbbox_y_max = bbbox.ymin + bbbox.height

                rect_start_point =mp_drawing._normalized_to_pixel_coordinates(
                  bbbox_x_min,
                  bbbox_y_min,
                  shape[1],
                  shape[0])

                rect_end_point = mp_drawing._normalized_to_pixel_coordinates(
                   bbbox_x_max,
                   bbbox_y_max,
                   shape[1], shape[0])
                crop_img = image[
                   rect_start_point[1]:rect_end_point[1],
                   rect_start_point[0]:rect_end_point[0]
                   ]
                yield crop_img,True
        else:
            return None, False


if __name__=='__main__':
    image_path = '../image_examples/COCO_train2014_000000000165.jpg'
    save_directory = '../image_examples/edges'
    i = 0
    image = cv2.imread(image_path)

    for face_image, success in detect_face_from_image(image):
        if success:
            cv2.imwrite(os.path.join(save_directory,'faces')+ str(i) + '.png', face_image)
            i+=1
