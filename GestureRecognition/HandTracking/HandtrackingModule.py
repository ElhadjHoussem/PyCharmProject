import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.5, trackingConfidence=0.5):
        self.mode = mode
        self.maxhands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxhands, self.detectionConfidence, self.trackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        img_flip = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img_flip, cv2.COLOR_BGR2RGB)

        imgRGB.flags.writeable = False

        self.results = self.hands.process(imgRGB)
        imgRGB.flags.writeable = True

        if self.results.multi_hand_landmarks:
            for handLMs in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img_flip,handLMs,self.mpHands.HAND_CONNECTIONS)
        return img_flip

    def findPosition(self,img,handNo=0,draw=True):
        lmList=[]
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNo]

            for id,lm in enumerate(hand.landmark):

                h,w,c = img.shape
                px,py = int(lm.x*w), int(lm.y*h)
                #print(id,px,py)
                lmList.append([id,px,py])
                if draw:
                    cv2.circle(img,(px,py),15,(255,0,255),cv2.FILLED)
        return lmList

def main():
    pTime=0
    cTime=0
    cap =cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList=detector.findPosition(img)
        if len(lmList)!=0:
            print(lmList[4])

        cTime =time.time()
        fps=1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv2.imshow("image",img)
        if cv2.waitKey(5) & 0xFF == 27:
          break
if __name__ =="__main__":
    main()


