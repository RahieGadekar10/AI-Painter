import handtrackingmodule as track
import time
import mediapipe as mp
import cv2
import os
import numpy as np

def main() :

    overlay = []
    canvas = np.zeros((720, 1280 ,3), np.uint8)
    drawcolor = (21, 39, 250)
    xp = 0
    yp = 0
    folder = 'images'
    dir = os.listdir('images')

    for img in dir:
     image = cv2.imread(f'{folder}/{img}')
     overlay.append(image)

    header = overlay[0]
    ptime = 0
    ctime = 0

    tracking= track.handtracking(maxHands=1)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

    while True :

        ret , frame = cap.read()
        frame = cv2.flip(frame , 180)
        frame = tracking.landmarks(frame)
        points = tracking.list_points(frame)

        if len(points) != 0 :

            x1 , y1 = points[8][0] , points[8][1]
            x2 , y2 = points[12][0] , points[12][1]

            if((points[12][1] > points[10][1]) and (points[8][1] < points[6][1])) :
                cv2.putText(frame, "Selection Mode", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 150))
                xp ,yp = 0,0

                if y1 < 125:

                    if 110 < x1 < 350:
                        header = overlay[0]
                        drawcolor = (21, 39, 250)

                    if 400 < x1 < 500:
                        header = overlay[1]
                        drawcolor = (238, 197, 88)

                    if 550 < x1 < 650:
                        header = overlay[2]
                        drawcolor = (47, 241, 247)

                    if 750 < x1 < 850:
                        header = overlay[3]
                        drawcolor = (35, 246, 128)

                    if 950 < x1 < 1000:
                        header = overlay[4]
                        drawcolor = (0, 0, 0)

                    if 1100 < x1:
                        header = overlay[5]
                        drawcolor = (255, 255, 255)

                cv2.circle(frame,(x1,y1), 15 , drawcolor, cv2.FILLED)


            if((points[12][1] < points[11][1]) and (points[8][1] < points[7][1])) :
                cv2.putText(frame , "Drawing Mode" , (10,250), cv2.FONT_HERSHEY_SIMPLEX ,1, (255, 255, 150))
                cv2.rectangle(frame, (x1, y1 + 10), (x2, y2 + 10), drawcolor, cv2.FILLED)

                if xp == 0 and yp == 0 :
                    xp , yp = x1 , y1

                if(drawcolor == (0, 0, 0)):
                    cv2.putText(frame, "Erase Mode", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 150))
                    cv2.line(frame, (xp, yp), (x1, y1), drawcolor, 80)
                    cv2.line(canvas, (xp, yp), (x1, y1), drawcolor, 80)
                    cv2.rectangle(frame, (x1, y1 + 10), (x2, y2 + 10), drawcolor, cv2.FILLED)
                else :
                    cv2.line(frame, (xp, yp), (x1, y1), drawcolor, 15)
                    cv2.line(canvas, (xp, yp), (x1, y1), drawcolor, 15)

                xp, yp = x1, y1

        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime

        cv2.putText(frame, str(int(fps)) , (10,200),cv2.FONT_HERSHEY_SIMPLEX , 2, (255, 255, 150), 1)
        frame[0:125 , 0:1280] = header

        imggray = cv2.cvtColor(canvas , cv2.COLOR_BGR2GRAY)
        ret , imginv = cv2.threshold(imggray ,50,255, cv2.THRESH_BINARY_INV)
        imginv = cv2.cvtColor(imginv , cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame , imginv)
        frame = cv2.bitwise_or(frame , canvas)

        cv2.putText(frame, "Press 'c' to Clear Screen", (950, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 150),2)
        if cv2.waitKey(1) & 0XFF == ord('c') :
            canvas = np.zeros((720, 1280, 3), np.uint8)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0XFF == ord('q') :
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__' :
    main()