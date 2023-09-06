#!/usr/bin/env python3

import cv2
import mediapipe as mp
import numpy as np
from anx_interface import TfliteInterface, DeviceType, Anx
from pynput.keyboard import Key, Controller
import time
import threading
import sys

max_num_hands = 2

rps_gesture = {0:'rock', 5:'paper', 9:'scissors'}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition model
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 10.0, (700, 520))

class Game:
    def __init__(self):
        self.state = 0
        self.is_running = False
        self.player_one = 0
        self.player_two = 0
        self.winner = ""
        self.rps_result = []
        self.t1 = threading.Thread(target=self.game_state)
        self.t2 = threading.Thread(target=self.play)
        self.keyboard = Controller()
        self.anx = Anx()
        self.anx.start_device_camera(fps=30, width=640, height=480, pixel_format=0)
        self.cap = self.anx.device_camera

    def get_winner(self,img):
        winner = None
        text = ''
        pos = 0
        if self.rps_result[0]['rps']=='rock':
            if self.rps_result[1]['rps']=='rock'     :self.winner = 'TIE'
            elif self.rps_result[1]['rps']=='paper'  : self.winner = 'PAPER WINS'  ; winner = 1 ; self.player_two += 1
            elif self.rps_result[1]['rps']=='scissors': self.winner = 'ROCK WINS'   ; winner = 0 ; self.player_one += 1
        elif self.rps_result[0]['rps']=='paper':
            if self.rps_result[1]['rps']=='rock'     : self.winner = 'PAPER WINS'  ; winner = 0 ; self.player_one += 1
            elif self.rps_result[1]['rps']=='paper'  : self.winner = 'TIE'
            elif self.rps_result[1]['rps']=='scissors': self.winner = 'SCISSORS WINS'; winner = 1 ; self.player_two += 1
        elif self.rps_result[0]['rps']=='scissors':
           if self.rps_result[1]['rps']=='rock'     : self.winner = 'ROCK WINS'   ; winner = 1 ; self.player_two += 1
           elif self.rps_result[1]['rps']=='paper'  : self.winner = 'SCISSORS WINS'; winner = 0 ; self.player_one += 1
           elif self.rps_result[1]['rps']=='scissors': self.winner = 'TIE'

        if winner is not None:
            print("\n ----SCORE----")
            print("\n P1:{} | P2:{}".format(self.player_one, self.player_two))
            #cv2.putText(img, text="WINNER", org=(self.rps_result[winner]['org'][0], self.rps_result[winner]['org'][1] + 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        #cv2.putText(img, text=text, org=(self.rps_result[pos]['org'][0] - 70, self.rps_result[pos]['org'][1] - 170), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
        self.state = 2
        return img

    def play(self):
        cv2.namedWindow('Game', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Game", 700, 520)
        cv2.moveWindow('Game', 1100, 400)
        blank_image = np.zeros(shape=[520, 700, 3], dtype=np.uint8)
        cv2.putText(blank_image, text = "Type y to continue", org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(0, 0, 255), thickness=2)
        cv2.imshow("Game", blank_image)
        cv2.waitKey(0)

        while self.is_running:
            if self.state == 0 or self.state == 2:
                continue
            success, img = self.cap.read()
            if not success:
                # print("PROBLEM")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            img = cv2.flip(img, 1)
            result = hands.process(img)
        
            count = 1
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                self.rps_result = []
                for res in result.multi_hand_landmarks:
                    if count > max_num_hands: break
                    joint = np.zeros((21, 3))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z]

                    # Compute angles between joints
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                    v = v2 - v1 # [20,3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree

                    # Inference gesture
                    data = np.array([angle], dtype=np.float32)
                    ret, results, neighbours, dist = knn.findNearest(data, 3)
                    idx = int(results[0][0])
                    text = "PLAYER {}".format(count)
                    # Draw gesture result
                    if idx in rps_gesture.keys():
                        org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))
                        cv2.putText(img, text=rps_gesture[idx].upper(), org=(org[0], org[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                        cv2.putText(img, text=text, org=(org[0] - 70, org[1] - 200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)

                        self.rps_result.append({
                        'rps': rps_gesture[idx],
                        'org': org
                        })
                        count += 1

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                if len(self.rps_result) >= 2:
                    img = self.get_winner(img)

                 
            p_one_img = img[:, :320]
            p_two_img = img[:, 320:]
            p_one_img= cv2.copyMakeBorder(p_one_img,20,20,20,0,cv2.BORDER_CONSTANT,value=[0,0,0])
            p_two_img= cv2.copyMakeBorder(p_two_img,20,20,20,20,cv2.BORDER_CONSTANT,value=[0,0,0])

            img = cv2.hconcat([p_one_img, p_two_img])
           
            out.write(img)  
            if self.state == 2:
                self.state = 0
                cv2.putText(img, text=self.winner, org=(260, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
                cv2.putText(img, text="Type y to continue playing", org=(260, 75), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2)
                cv2.putText(img, text=str(self.player_one), org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                cv2.putText(img, text=str(self.player_two), org=(650, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                cv2.imshow("Game", img)
                cv2.waitKey(0)
                continue

            else: cv2.imshow('Game', img)
            if cv2.waitKey(1) == ord('q'):
                break

        out.release()
        self.anx.stop_device_camera()
        cv2.destroyAllWindows()

    def game_state(self):
            while self.is_running:
                if self.state == 0:
                    ch = input("Ready to play? [y/n]: ")
                    if ch == "y": time.sleep(5);self.keyboard.press(Key.esc);self.keyboard.release(Key.esc);self.state = 1; 
                    elif ch == "n":
                        print("Exiting ..")
                        time.sleep(3);self.keyboard.press(Key.esc);self.keyboard.release(Key.esc)
                        self.is_running = False
                        exit(1)
         

    def run(self):
        self.is_running = True
        self.t1.start()
        self.t2.start()


    def wait(self):
        self.t1.join()
        self.t2.join()  

                
if __name__ == "__main__":
     game = Game()
     game.run()
     game.wait()

