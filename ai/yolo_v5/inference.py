#!/usr/bin/env python3

import sys
import time
import cv2
import numpy as np
from anx_interface import TfliteInterface, DeviceType, Anx
import click


# Visualization parameters
_BOX_THRESHOLD = 0.5
_CLASS_THRESHOLD = 0.5
_IMG_WIDTH = 640
_IMG_HEIGHT = 480
_LABEL_SIZE = 0.6

@click.command()
@click.option("--video-src", help="Specifiy video source (device_camera/usb_camera/video).", required = False, default = "device_camera")
@click.option("--dev", help="USB camera.", required = False, default = "/dev/video3")
@click.option("--video-path", help="Path to video file.", required = False, default = "office.mp4")
@click.option("--model", help="Path to model file.", required = False, default = "yolo_v5.tflite")
@click.option("--label-map", help="Path to label map text file.", required = False, default = "labelmap.txt")
@click.option("--width", help="Width of frame to capture from camera.", required = False, default = 640)
@click.option("--height", help="Height of frame to capture from camera.", required = False, default = 480)
def run(video_src, dev, video_path, model, label_map, width, height) -> None:

  """
  Continuously run YOLOv5 inference on images acquired from the video stream.
  """

  # use DeviceType.CPU to inference on CPU
  tflite_interface = TfliteInterface(DeviceType.GPU) 
  tflite_interface.load_model(model)

  input_details = tflite_interface.input_tensors
  output_details = tflite_interface.output_tensors

  # only one input tensor
  # yolo v5 expects rgb frames of 320, 320
  in_height = input_details[0].shape[1]
  in_width = input_details[0].shape[2]

  with open(label_map, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
    
  colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')


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

  start_time = time.time()

  while True:
    success, frame = cap.read()
    if not success:
       continue

    pad = round(abs( _IMG_WIDTH - _IMG_HEIGHT) / 2)
    x_pad = pad if _IMG_HEIGHT > _IMG_WIDTH else 0
    y_pad = pad if _IMG_WIDTH > _IMG_HEIGHT else 0
    frame_padded = cv2.copyMakeBorder(frame, top=y_pad, bottom=y_pad, left=x_pad, right=x_pad,
                                        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    IMG_HEIGHT, IMG_WIDTH = frame_padded.shape[:2]
    frame_resized = cv2.resize(frame_padded, (in_width, in_height))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(frame_rgb/255, axis=0).astype('float32')

    tflite_interface.set_inputs([input_data])
    start = time.time()
    tflite_interface.invoke()
    end = time.time()
    dt = (end-start)*1000
    print(f"dt : {dt}")
    output_data = tflite_interface.get_output()
        
    outputs = np.squeeze(output_data[0])
    boxes = []
    box_confidences = []
    classes = []
    class_probs = []
        
    for output in outputs:
      box_confidence = output[4]
      if box_confidence < _BOX_THRESHOLD:
        continue
        
      class_ = output[5:].argmax(axis=0)
      class_prob = output[5:][class_]
        
      if class_prob < _CLASS_THRESHOLD:
        continue

      cx, cy, w, h = output[:4] * np.array([IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH, IMG_HEIGHT])
      x = round(cx - w / 2)
      y = round(cy - h / 2)
      w, h = round(w), round(h)
        
      boxes.append([x, y, w, h])
      box_confidences.append(box_confidence)
      classes.append(class_)
      class_probs.append(class_prob)

    indices = cv2.dnn.NMSBoxes(boxes, box_confidences, _BOX_THRESHOLD, _BOX_THRESHOLD - 0.1)
    p_count = 0
    for index in indices:
      x, y, w, h = boxes[index]
      class_name = labels[classes[index]]
      score = box_confidences[index] * class_probs[index]
      color = [int(c) for c in colors[classes[index]]]
      text_color = (255, 255, 255) if sum(color) < 144 * 3 else (0, 0, 0)
        
      cv2.rectangle(frame_padded, (x, y), (x + w, y + h), color, 2)
      label = f'{class_name}: {score*100:.2f}%'
      labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, _LABEL_SIZE, 2)
      cv2.rectangle(frame_padded, (x, y + baseLine), (x + labelSize[0], y - baseLine - labelSize[1]), color, cv2.FILLED) 
      cv2.putText(frame_padded, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, _LABEL_SIZE, text_color, 1)

    img_show = frame_padded[y_pad: IMG_HEIGHT - y_pad, x_pad: IMG_WIDTH - x_pad]
    img_show = cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR) if video_src == "device_camera" else img_show
    inference_time = "{0:.3f}".format((end-start)*1000)
    time_label = f'dt : {inference_time} ms'
    fps = int(1/(end-start))
    fps_label =  f'FPS : {fps}'
    cv2.putText(img_show, time_label,(480 , 40), cv2.FONT_HERSHEY_SIMPLEX, _LABEL_SIZE, (255,255,255), 2)
    cv2.putText(img_show, fps_label,(500  , 60), cv2.FONT_HERSHEY_SIMPLEX, _LABEL_SIZE, (255,255,255), 2)
    cv2.imshow('Object detection', img_show)
    if cv2.waitKey(1) == ord('q'):
      break

    out.write(img_show)  
  
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
