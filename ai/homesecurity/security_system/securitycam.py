#!/usr/bin/env python3

import cv2, time, os, signal
import queue
import numpy as np
import sys
from anx_interface import TfliteInterface, DeviceType, Anx
sys.path.insert(0, '/homesecurity/utils')
from config import *
from file_manager import FileManagerThread, counter

file_q = queue.Queue()

os.environ['TZ']= location
time.tzset()

print(('Time: {}. Starting...'.format(time.strftime("%a, %d %b %Y %H:%M:%S",time.localtime()))))
print('Finished importing modules')
print(('Sleeping for {} seconds'.format(initial_sleep)))
time.sleep(initial_sleep)

anx = Anx()
anx.start_device_camera(fps=30,width=640,height=480,pixel_format=0)
camera = anx.device_camera

time_since_last_sent=200 #in minutes
time_last_sent=time.time()
file_counter=counter(countfile)

video_num=file_counter.count

fmt= FileManagerThread(h264_q=file_q)
fmt.start()

def record_for(rec_time):
    video_num=file_counter.get_current_count()
    this_file='video{}.mp4'.format(video_num)
    if this_file in os.listdir(h264_folder):
        os.remove(h264_folder+this_file)
    #remove this file, if it already exists

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(h264_folder+this_file, fourcc, 20.0, (640, 480))

    rec_time = rec_time*25

    print('Video recording started')

    while rec_time>0:
        success, frame = camera.read()
        if not success:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
        rec_time -= 1

    out.release()

    file_counter.update_count()
    file_q.put(this_file)
    #handle_file(filename)
    return

def handler(signum, frame):
    fmt.join(0.05)
    exit(1)

if __name__ == '__main__':

    signal.signal(signal.SIGINT, handler)
    start=time.time()
    not_beginning=False
    count=0
    rec_time = 5
    fgbg=cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=8, detectShadows=False)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    while True:
        success, rawCapture=camera.read()
        if not success:
            continue
        frame = cv2.flip(rawCapture, 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        fgmask=fgbg.apply(frame,learningRate=0.1)
        fgmask=cv2.erode(fgmask,kernel,iterations=1)
        #Detecting motion
        motion_magnitude=np.mean(fgmask)

        count+=1
        time_since_last_sent=(time.time()-time_last_sent)/60

        if count%50==0:
            #Start after first 50 frames
            not_beginning=True
            print(('frame rate={}'.format(count/(time.time()-start))))

        if not_beginning and motion_magnitude>=motion_threshold:
            print('Motion detected!!')
            record_for(rec_time)
            not_beginning=False
            count=0
            fgbg=cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=8, detectShadows=False)
            start=time.time()

        elif count>1000000:
            print('count is 1 million. Setting count to zero')
            count=0
       
    anx.wait()
    
