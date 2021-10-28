import cv2
import time
import os
import HandtrackingModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


pTime = 0

detector = htm.HandDetector(detectionConfidence=0.7)

fingerTipIds = [4, 8, 12, 16, 20]

while True:
    success, img_ = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    img = detector.findHands(img_)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers = []

        # Thumb
        if lmList[fingerTipIds[0]][1] < lmList[fingerTipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if lmList[fingerTipIds[id]][2] < lmList[fingerTipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)


        cv2.rectangle(img, (20, 20), (130, 220), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (35, 170), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    7, (255, 0, 0), 10)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
