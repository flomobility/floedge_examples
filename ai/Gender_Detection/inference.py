#!/usr/bin/env python3

from anx_interface import TfliteInterface, DeviceType, Anx
import numpy as np
import cv2
from time import time
import click
import sys

string_pred_gen = ['Female', 'Male']

@click.command()
@click.option("--video-src", help="Specifiy video source (device_camera/usb_camera/video).", required = False, default = "device_camera")
@click.option("--dev", help="USB camera.", required = False, default = "/dev/video3")
@click.option("--video-path", help="Path to video file.", required = False, default = "UPSC_Sample.mp4")
@click.option("--model", help="Path to model file.", required = False, default = "GenderClass_06_03-20-08.tflite")
@click.option("--classifier", help="Path to classifier xml file.", required = False, default = "haarcascade_frontalface_default.xml")
@click.option("--width", help="Width of frame to capture from camera.", required = False, default = 640)
@click.option("--height", help="Height of frame to capture from camera.", required = False, default = 480)
def run(video_src, dev, video_path, model, classifier, width, height) -> None:

  """
  Continuously run face detection and gender classification model stack 
  inference on images acquired from the video stream.
  """
  
  face_cascade = cv2.CascadeClassifier(classifier)

  # use DeviceType.CPU to inference on CPU
  tflite_interface_gender = TfliteInterface(DeviceType.GPU) 
  tflite_interface_gender.load_model(model)

  input_details = tflite_interface_gender.input_tensors
  output_details = tflite_interface_gender.output_tensors

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))
  
  if video_src == "video":
    cap = cv2.VideoCapture(video_path)
    #Set width and height here or pass as argument
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
  
  elif video_src == "device_camera":
    anx = Anx()
    anx.start_device_camera(fps=30, width=640, height=480, pixel_format=0)
    cap = anx.device_camera

  elif video_src == "usb_camera":
    cap = cv2.VideoCapture(dev)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
  
  else:
    sys.exit("\nInvalid camera ID or video file path. Use --help to know more.")

  while True:
    font = cv2.FONT_HERSHEY_SIMPLEX
    time_start = time()
    success, frame = cap.read()
    
    if not success:
        continue
        
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    start = time()
    end = time()
    prev_fps=""
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor = 1.2, minNeighbors=5)
    for x,y,w,h in faces:
        saved_image = frame          
        input_im = saved_image[y:y+h, x:x+w]
        
        if input_im is None:
            print("Nu a fost detectata nicio fata")
        else:
            input_im = cv2.resize(input_im, (224,224))
            input_im = input_im.astype('float')
            input_im = input_im / 255
            input_im = np.array(input_im)
            input_im = np.expand_dims(input_im, axis = 0)

            # Predict
            input_data = np.array(input_im, dtype=np.float32)
    
            tflite_interface_gender.set_inputs([input_data])
            start = time()
            tflite_interface_gender.invoke()
            end = time()
            
            print((end-start)*1000)
            output_data_gender = tflite_interface_gender.get_output()
            index_pred_gender = int(np.argmax(output_data_gender))
            prezic_gender = string_pred_gen[index_pred_gender]
            
            cv2.putText(frame,prezic_gender, (x,y), font, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255), 1)

    if (round(1.0 / (end-start), 2)<200):
        fps = "FPS: " + str(round(1.0 / (end-start), 2))
        print((1.0/end-start))
        prev_fps = fps
    else:
        fps = prev_fps
    cv2.putText(frame, fps, (1000,40), font, 1, (250,250,250), 3, cv2.LINE_AA) 
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if video_src == "device_camera" else frame       
    cv2.imshow("Detecting faces...", frame)
    
    
    key = cv2.waitKey(1)

    if key == ord('q') or key == ord('Q'):
        break

    out.write(frame)  
  
  try:
    cap._stop()
    cap.release()
  except:
    print("Terminating")

  out.release()
  cv2.destroyAllWindows()
  tflite_interface_gender.unload_model()


if __name__ == '__main__':
  run()
