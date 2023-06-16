#!/usr/bin/env python3

import cv2
import numpy as np
import mediapipe as mp
import alsaaudio

import math
import time

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

mixer = alsaaudio.Mixer()
volume = mixer.getvolume()[0]
minVol, maxVol = 0, 100

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x1, y1 = int(hand_landmarks.landmark[4].x * wCam), int(hand_landmarks.landmark[4].y * hCam)
                x2, y2 = int(hand_landmarks.landmark[8].x * wCam), int(hand_landmarks.landmark[8].y * hCam)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                length = math.hypot(x2 - x1, y2 - y1)

                vol = np.interp(length, [50, 300], [minVol, maxVol])
                volBar = np.interp(length, [50, 300], [400, 150])
                volPer = np.interp(length, [50, 300], [0, 100])
                print(int(length), vol)

                mixer.setvolume(int(vol))

                if length < 50:
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

