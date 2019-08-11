import numpy as np
from collections import deque
import cv2 as cv



pts = deque(maxlen=512)
blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
blackboard_copy = np.zeros((480, 640, 3), dtype=np.uint8)
digit = np.zeros((200, 200, 3), dtype=np.uint8)
pred_class = 0
cap = cv.VideoCapture(0)
while (cap.isOpened()):
        ret, img = cap.read()
        img = cv.flip(img, 1)
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)
        Lower_green = np.array([110 ,50, 50])
        Upper_green = np.array([130, 255, 255])
        
        mask = cv.inRange(hsv, Lower_green, Upper_green)
        mask = cv.erode(mask, kernel, iterations=2)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        # mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
        mask = cv.dilate(mask, kernel, iterations=1)
        res = cv.bitwise_and(img, img, mask=mask)
        cnts, heir = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2:]
        center = None

        if len(cnts) >= 1:
            cnt = max(cnts, key=cv.contourArea)
            if cv.contourArea(cnt) > 200:
                ((x, y), radius) = cv.minEnclosingCircle(cnt)
                cv.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv.circle(img, center, 5, (0, 0, 255), -1)
                M = cv.moments(cnt)
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                pts.appendleft(center)
                for i in range(1, len(pts)):
                    if pts[i - 1] is None or pts[i] is None:
                        continue
                    cv.line(blackboard, pts[i - 1], pts[i], (255, 255, 255), 7)
                    cv.line(img, pts[i - 1], pts[i], (0, 0, 255), 2)
        
        cv.imshow("Frame", img)
        cv.imshow("black", blackboard_copy)
        k = cv.waitKey(10)
        if k == 27:
            break
    
cap.release()
cv.destroyAllWindows()    