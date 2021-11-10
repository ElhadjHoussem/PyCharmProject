import cv2
import mediapipe as mp
import numpy as np
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# For static images:
IMAGE_FILES = []
with mp_face_detection.FaceDetection( model_selection=1, min_detection_confidence=0.5) as face_detection:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw face detections of each face.
    if not results.detections:
      continue
    annotated_image = image.copy()
    for detection in results.detections:
      print('Nose tip:')
      print(mp_face_detection.get_key_point(
          detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
      mp_drawing.draw_detection(annotated_image, detection)
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    shape = image.shape
    if results.detections:
      for detection in results.detections:
        offset_x =0.05
        offset_y =0.2
        bbbox = detection.location_data.relative_bounding_box
        rect_start_point =mp_drawing._normalized_to_pixel_coordinates(
          bbbox.xmin-offset_x if  bbbox.xmin-offset_x>0 else bbbox.xmin,
          bbbox.ymin-offset_y if  bbbox.ymin-offset_y>0 else bbbox.ymin,
          shape[1],
          shape[0])

        rect_end_point = mp_drawing._normalized_to_pixel_coordinates(
           bbbox.xmin+offset_x + bbbox.width if bbbox.xmin+offset_x+ bbbox.width <1 else bbbox.xmin + bbbox.width,
           bbbox.ymin + bbbox.height,
           shape[1], shape[0])
        crop_img = image[
           rect_start_point[1]:rect_end_point[1],
           rect_start_point[0]:rect_end_point[0]
           ]
        cv2.rectangle(image, rect_start_point, rect_end_point,(255,0,0), 2)


    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    cv2.imshow('crop Face Detection', cv2.flip(crop_img, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()