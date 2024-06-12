#gray cam and also shows fram size with no RGB channels
import numpy as np
import cv2
address=0
cap = cv2.VideoCapture(address)
#cap2 = cv2.VideoCapture(address)

while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    print(frame.shape)

    cv2.imshow('frame', frame)
    #cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()