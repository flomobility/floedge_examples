#!/usr/bin/env python3

import click
import sys
import time
from typing import List
from anx_interface import TfliteInterface, DeviceType, Anx

import cv2
import numpy as np
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

# Visualization parameters
_FPS_AVERAGE_FRAME_COUNT = 10
_FPS_LEFT_MARGIN = 24  # pixels
_LEGEND_TEXT_COLOR = (0, 0, 255)  # red
_LEGEND_BACKGROUND_COLOR = (255, 255, 255)  # white
_LEGEND_FONT_SIZE = 1
_LEGEND_FONT_THICKNESS = 1
_LEGEND_ROW_SIZE = 20  # pixels
_LEGEND_RECT_SIZE = 16  # pixels
_LABEL_MARGIN = 10
_OVERLAY_ALPHA = 0.5
_PADDING_WIDTH_FOR_LEGEND = 150  # pixels

@click.command()
@click.option("--video-src", help="Specifiy video source (device_camera/usb_camera/video).", required = False, default = "device_camera")
@click.option("--dev", help="USB camera.", required = False, default = "/dev/video3")
@click.option("--video-path", help="Path to video file.", required = False, default = "cars.mp4")
@click.option("--model", help="Path to model file.", required = False, default = "lite-model_deeplabv3_1_metadata_2.tflite")
@click.option("--display-mode", help="Display mode of segmentation (overlay/side-by-side).", required = False, default = "overlay")
@click.option("--num-threads", help="Number of CPU threads to run the model.", required = False, default = 4)
@click.option("--tpu-flag", help="To run model on Edge TPU.", required = False, default = False)
@click.option("--width", help="Width of frame to capture from camera.", required = False, default = 640)
@click.option("--height", help="Height of frame to capture from camera.", required = False, default = 480)
def run(video_src, dev, video_path, model, display_mode, num_threads, tpu_flag, width, height) -> None:
  """
  Continuously run DeepLabV3 sgementation inference on images acquired from the video stream.
  Visualize segmentation result on image.
  """

  # Initialize the image segmentation model.
  base_options = core.BaseOptions(
      file_name=model, use_coral=tpu_flag, num_threads=num_threads)
  segmentation_options = processor.SegmentationOptions(
      output_type=processor.OutputType.CATEGORY_MASK)
  options = vision.ImageSegmenterOptions(
      base_options=base_options, segmentation_options=segmentation_options)

  segmenter = vision.ImageSegmenter.create_from_options(options)

  # Variables to calculate FPS
  counter, fps = 0, 0

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width + _PADDING_WIDTH_FOR_LEGEND, height))
  
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

  # Continuously capture images from the camera and run inference.
  while True:
    success, image = cap.read()
    if not success:
      continue

    counter += 1
    image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create TensorImage from the RGB image
    tensor_image = vision.TensorImage.create_from_array(rgb_image)

    # Segment with each frame from camera.
    segmentation_result = segmenter.segment(tensor_image)


    # Convert the segmentation result into an image.
    seg_map_img, found_colored_labels = utils.segmentation_map_to_image(segmentation_result)
    # Resize the segmentation mask to be the same shape as input image.
    seg_map_img = cv2.resize(
        seg_map_img,
        dsize=(image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST)

    # Visualize segmentation result on image.
    overlay = visualize(image, seg_map_img, display_mode, fps,
                        found_colored_labels)
    
    overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR) if video_src == "device_camera" else overlay
    # Calculate the FPS
    if counter % _FPS_AVERAGE_FRAME_COUNT == 0:
      end_time = time.time()
      fps = _FPS_AVERAGE_FRAME_COUNT / (end_time - start_time)
      start_time = time.time()

    # Stop the program if the Q key is pressed.
    if cv2.waitKey(1) == ord("q"):
      break
    cv2.imshow('image_segmentation', overlay)

    out.write(overlay)  
  
  try:
    cap._stop()
    cap.release()
  except:
    print("Terminating")
 
  out.release()
  cv2.destroyAllWindows()


def visualize(input_image: np.ndarray, segmentation_map_image: np.ndarray,
              display_mode: str, fps: float,
              colored_labels: List[processor.ColoredLabel]) -> np.ndarray:
  """
  Visualize segmentation result on image.
  """
  # Show the input image and the segmentation map image.
  if display_mode == 'overlay':
    # Overlay mode.
    overlay = cv2.addWeighted(input_image, _OVERLAY_ALPHA,
                              segmentation_map_image, _OVERLAY_ALPHA, 0)
  elif display_mode == 'side-by-side':
    # Side by side mode.
    overlay = cv2.hconcat([input_image, segmentation_map_image])
  else:
    sys.exit(f'ERROR: Unsupported display mode: {display_mode}.')

  # Show the FPS
  fps_text = 'FPS = ' + str(int(fps))
  text_location = (_FPS_LEFT_MARGIN, _LEGEND_ROW_SIZE)
  cv2.putText(overlay, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
              _LEGEND_FONT_SIZE, _LEGEND_TEXT_COLOR, _LEGEND_FONT_THICKNESS)

  # Initialize the origin coordinates of the label.
  legend_x = overlay.shape[1] + _LABEL_MARGIN
  legend_y = overlay.shape[0] // _LEGEND_ROW_SIZE + _LABEL_MARGIN

  # Expand the frame to show the label.
  overlay = cv2.copyMakeBorder(overlay, 0, 0, 0, _PADDING_WIDTH_FOR_LEGEND,
                               cv2.BORDER_CONSTANT, None,
                               _LEGEND_BACKGROUND_COLOR)

  # Show the label on right-side frame.
  for colored_label in colored_labels:
    rect_color = colored_label.color
    start_point = (legend_x, legend_y)
    end_point = (legend_x + _LEGEND_RECT_SIZE, legend_y + _LEGEND_RECT_SIZE)
    cv2.rectangle(overlay, start_point, end_point, rect_color,
                  -_LEGEND_FONT_THICKNESS)

    label_location = legend_x + _LEGEND_RECT_SIZE + _LABEL_MARGIN, legend_y + _LABEL_MARGIN
    cv2.putText(overlay, colored_label.category_name, label_location,
                cv2.FONT_HERSHEY_PLAIN, _LEGEND_FONT_SIZE, _LEGEND_TEXT_COLOR,
                _LEGEND_FONT_THICKNESS)
    legend_y += (_LEGEND_RECT_SIZE + _LABEL_MARGIN)

  return overlay



if __name__ == '__main__':
  run()
