"""
Face Tracking Drone with OpenCV and DJI Tello (Dev File)
---------------------------------------------

This program connects to a DJI Tello drone and uses OpenCV to detect and track a human face in real-time.
It implements a basic PID controller to keep the face centered in the video frame and maintain a consistent
distance from the subject by adjusting the drone’s yaw and forward/backward motion.

Features:
- Real-time face detection using Haar cascades
- Autonomous takeoff and landing
- PID-based control for smooth face tracking
- Visual feedback with bounding boxes and tracking dot
- Adjustable distance control using face area thresholds

Press 'q' to safely land the drone and exit the program.

Requirements:
- djitellopy
- OpenCV
- NumPy

Author: [Jylen Tate]
Date: [2025-16-07]
"""

from djitellopy import Tello
import cv2
import numpy as np
import time

drone = Tello()
drone.connect()

drone.streamon()
drone.takeoff()
drone.send_rc_control(0,0,25,0)
time.sleep(2.2)

fbRange = [6200, 6800]
pid = [0.4, 0.4, 0]
pError = 0
w, h = 360, 240

def find_face(img):
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, scaleFactor=1.05, minNeighbors=5)

    myFacesList = []
    myFacesListArea = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x + w, x + h), (0, 0 , 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        myFacesList.append([cx, cy])
        myFacesListArea.append(area)
    if len(myFacesListArea) != 0:
        i = myFacesListArea.index(max(myFacesListArea))
        return img, [myFacesList[i], myFacesListArea[i]]
    else:
        return img, [[0,0], 0]

def track_face(drone, info, w, pid, pError):
    
    area = info[1]
    x, y = info[0]
    fb = 0
    error = x - w // 2
    speed = pid[0] * error + pid[1] * (error - pError)
    speed = int(np.clip(speed, -100, 100))

    if area > fbRange[0] and area < fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        fb = -20
    elif area < fbRange[0] and area != 0:
        fb = 20
    
    if x == 0:
        speed = 0
        error = 0
    
    drone.send_rc_control(0, fb, 0, speed)
    
    return error

while True:
    img = drone.get_frame_read().frame
    img = cv2.resize(img, (w,h))
    img, info = find_face(img)
    pError = (drone, info, pid, pError)
    cv2.imshow("Output", img)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        drone.land()
        break