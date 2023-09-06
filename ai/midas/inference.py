#!/usr/bin/env python3

import click
import numpy as np
import cv2
import time
import sys
from anx_interface import TfliteInterface, DeviceType, Anx
import threading

IMG_WIDTH = 640
IMG_HEIGHT = 480
LABEL_SIZE = 0.6

@click.command()
@click.option("--video-src", help="Specifiy video source (device_camera/usb_camera/video).", required = False, default = "device_camera")
@click.option("--dev", help="USB camera.", required = False, default = "/dev/video3")
@click.option("--video-path", help="Path to video file.", required = False, default = "midasdepth.mp4")
@click.option("--model", help="Path to model file.", required = False, default = "midas.tflite")
@click.option("--label-map", help="Path to label map text file.", required = False, default = "labelmap.txt")
@click.option("--width", help="Width of frame to capture from camera.", required = False, default = 640)
@click.option("--height", help="Height of frame to capture from camera.", required = False, default = 480)
def run(video_src, dev, video_path, model, label_map, width, height) -> None:

  """
  Continuously run MiDaS inference on images acquired from the video stream.
  """
  
  # use DeviceType.CPU to inference on CPU
  tflite_interface = TfliteInterface(DeviceType.GPU) 
  tflite_interface.load_model(model)

  input_details = tflite_interface.input_tensors
  output_details = tflite_interface.output_tensors

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width*2, height))
  
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


  start_time = time.time()

  while True:
    success, img = cap.read()
    if not success:
       continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input = np.float32(img) / 255
    width,height,_ = input.shape
    input = cv2.resize(input, (256, 256))
    input = np.expand_dims(input, axis=0)
	
    tflite_interface.set_inputs([input])
    start = time.time()
    tflite_interface.invoke()
    end = time.time()
        
    print("dt: {}".format((end-start)*1000))
    i_time = "{0:.3f}".format((end-start)*1000)
    time_label = 'Time :' + i_time + ' ms'
    fps_label =  'FPS :'+str(int(1/(end-start)))
      
    output = tflite_interface.get_output()
    output = (output-np.min(output))/(np.max(output)-np.min(output)) * 255
    output = np.uint8(output)
    output = np.squeeze(output)
    output =cv2.resize(output,(height,width))
    new_out = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
    combined_image = np.hstack((img,new_out))
    cv2.putText(combined_image, time_label,(100 , 40), cv2.FONT_HERSHEY_SIMPLEX, LABEL_SIZE, (255,255,255), 2)
    cv2.putText(combined_image, fps_label,(100  , 60), cv2.FONT_HERSHEY_SIMPLEX, LABEL_SIZE, (255,255,255), 2)
    combined_image = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR) if video_src != "device_camera" else combined_image
    cv2.imshow('Output', combined_image)

    if cv2.waitKey(1) == ord('q'):
      break  

    out.write(combined_image)  
  
  try:
    cap._stop()
    cap.release()
  except:
    print("Terminating")

  out.release()
  cv2.destroyAllWindows()
  tflite_interface.unload_model()

if __name__ == '__main__':
  run()
