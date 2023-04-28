#!/usr/bin/python3

import time

import cv2
import numpy as np
from anx_interface import TfliteInterface, DeviceType

TF_LITE_MODEL = './yolo_v5.tflite'
LABEL_MAP = './labelmap.txt'
BOX_THRESHOLD = 0.5
CLASS_THRESHOLD = 0.5

IMG_WIDTH = 640
IMG_HEIGHT = 360
LABEL_SIZE = 0.6

# use DeviceType.CPU to inference on CPU
tflite_interface = TfliteInterface(DeviceType.GPU) 
tflite_interface.load_model(TF_LITE_MODEL)

input_details = tflite_interface.input_tensors
output_details = tflite_interface.output_tensors

# only one input tensor
# yolo v5 expects rgb frames of 320, 320
height = input_details[0].shape[1]
width = input_details[0].shape[2]

with open(LABEL_MAP, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
    
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

cap = cv2.VideoCapture('./cars.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)

while cap.isOpened():
    success, frame = cap.read()
    pad = round(abs(IMG_WIDTH - IMG_HEIGHT) / 2)
    x_pad = pad if IMG_HEIGHT > IMG_WIDTH else 0
    y_pad = pad if IMG_WIDTH > IMG_HEIGHT else 0
    frame_padded = cv2.copyMakeBorder(frame, top=y_pad, bottom=y_pad, left=x_pad, right=x_pad,
                                    borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    IMG_HEIGHT, IMG_WIDTH = frame_padded.shape[:2]
    
    frame_rgb = cv2.cvtColor(frame_padded, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized/255, axis=0).astype('float32')
    
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
        if box_confidence < BOX_THRESHOLD:
            continue
    
        class_ = output[5:].argmax(axis=0)
        class_prob = output[5:][class_]
    
        if class_prob < CLASS_THRESHOLD:
            continue

        cx, cy, w, h = output[:4] * np.array([IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH, IMG_HEIGHT])
        x = round(cx - w / 2)
        y = round(cy - h / 2)
        w, h = round(w), round(h)
    
        boxes.append([x, y, w, h])
        box_confidences.append(box_confidence)
        classes.append(class_)
        class_probs.append(class_prob)

    indices = cv2.dnn.NMSBoxes(boxes, box_confidences, BOX_THRESHOLD, BOX_THRESHOLD - 0.1)
    p_count = 0
    for index in indices:
        x, y, w, h = boxes[index]
        class_name = labels[classes[index]]
        score = box_confidences[index] * class_probs[index]
        color = [int(c) for c in colors[classes[index]]]
        text_color = (255, 255, 255) if sum(color) < 144 * 3 else (0, 0, 0)
    
        cv2.rectangle(frame_padded, (x, y), (x + w, y + h), color, 2)
        label = f'{class_name}: {score*100:.2f}%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SIZE, 2)
        cv2.rectangle(frame_padded,
                      (x, y + baseLine), (x + labelSize[0], y - baseLine - labelSize[1]),
                      color, cv2.FILLED) 
        cv2.putText(frame_padded, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, LABEL_SIZE, text_color, 1)

    img_show = frame_padded[y_pad: IMG_HEIGHT - y_pad, x_pad: IMG_WIDTH - x_pad]
    inference_time = "{0:.3f}".format((end-start)*1000)
    time_label = f'dt : {inference_time} ms'
    fps = int(1/(end-start))
    fps_label =  f'FPS : {fps}'
    cv2.putText(img_show, time_label,(480 , 40), cv2.FONT_HERSHEY_SIMPLEX, LABEL_SIZE, (255,255,255), 2)
    cv2.putText(img_show, fps_label,(500  , 60), cv2.FONT_HERSHEY_SIMPLEX, LABEL_SIZE, (255,255,255), 2)
    cv2.imshow('Object detection', img_show)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
tflite_interface.unload_model()
