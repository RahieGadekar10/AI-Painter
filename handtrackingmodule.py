import time
import mediapipe as mp
import cv2
import os
import numpy as np

class handtracking :

    def __init__(self , mode = False , maxHands = 2 , detectionCon = 0.8 , TrackingCon = 0.8):

        self.mode = mode
        self.maxHands = maxHands
        self.DetectionCon = detectionCon
        self.TrackingCon = TrackingCon

        self.mphands =  mp.solutions.hands
        self.hand = self.mphands.Hands(self.mode , self.maxHands , self.DetectionCon , self.TrackingCon)
        self.mpdraw = mp.solutions.drawing_utils

    def landmarks(self, frame):
        try :
            frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
            results = self.hand.process(frame)
            frame = cv2.cvtColor(frame , cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks :
                for hand in results.multi_hand_landmarks :
                    self.mpdraw.draw_landmarks(frame , hand , self.mphands.HAND_CONNECTIONS)
            return frame
        except Exception as e:
            print("Exception occured  : {}".format(e))


    def list_points(self, frame):
        points = []
        try :
            results = self.hand.process(frame)
            if results.multi_hand_landmarks :
                for hand in results.multi_hand_landmarks :
                    #self.mpdraw.draw_landmarks(frame , hand , self.mphands.HAND_CONNECTIONS)
                    for idx , idy in enumerate(hand.landmark) :
                        h , w , c = frame.shape
                        cx , cy = int(idy.x * w) , int(idy.y * h)
                        points.append([cx , cy])
            return points
        except Exception as e:
            print("Exception occured  : {}".format(e))

    def draw_points(self, frame):
        try :
            results = self.hand.process(frame)
            if results.multi_hand_landmarks :
                for hand in results.multi_hand_landmarks :
                    for idx , idy in enumerate(hand.landmark) :
                        h , w , c = frame.shape
                        cx , cy = int(idy.x * w) , int(idy.y * h)
                        cv2.circle(frame,(cx , cy) , 10  , (255,0,0), cv2.FILLED)
            return frame
        except Exception as e:
            print("Exception occured  : {}".format(e))

def main():

     tracking= handtracking()
     cap = cv2.VideoCapture(0)
     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
     ptime = 0
     ctime = 0

     while True :
        try :
            ret , frame = cap.read()
            frame = cv2.flip(frame , 180)
            frame = tracking.landmarks(frame)
            points = tracking.list_points(frame)

            # if len(points) != 0 :
            #     print(points)

            #frame = tracking.draw_points(frame)

            ctime = time.time()
            fps = 1/(ctime-ptime)
            ptime = ctime

            cv2.putText(frame, str(int(fps)) , (10,100),cv2.FONT_HERSHEY_SIMPLEX , 2, (255, 255, 150), 1)

            cv2.imshow("frame", frame)

        except Exception as e:
            print("Exception occured  : {}".format(e))

        if cv2.waitKey(1) & 0XFF == ord('q') :
            break

     cap.release()
     cv2.destroyAllWindows()

if __name__ == '__main__' :
    main()