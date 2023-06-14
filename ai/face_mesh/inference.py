#!/usr/bin/env python

import cv2
import mediapipe as mp
import time
import click
import sys

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

from anx_interface import TfliteInterface, DeviceType, Anx

@click.command()
@click.option("--video-src", help="Specifiy video source (device_camera/usb_camera/video).", required = False, default = "device_camera")
@click.option("--dev", help="USB camera.", required = False, default = "/dev/video3")
@click.option("--video-path", help="Path to video file.", required = False, default = "face.mp4")
@click.option("--width", help="Width of frame to capture from camera.", required = False, default = 640)
@click.option("--height", help="Height of frame to capture from camera.", required = False, default = 480)
def run(video_src, dev, video_path, width, height) -> None:

  """
  Continuously run face mesh inference on images acquired from the video stream.
  """

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

  drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

  with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    while True:
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      start = time.time()
      results = face_mesh.process(image)
      end = time.time()
      print(f"dt: {(end-start)*1000}")
      i_time = "{0:.3f}".format((end-start)*1000)
 
      # Draw the face mesh annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if video_src != "device_camera" else image
      if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
          mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
          mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
          mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
      # Flip the image horizontally for a selfie-view display.
      time_label = 'Time :' + i_time + ' ms'
      fps_label =  'FPS :'+str(int(1/(end-start)))
    
      cv2.putText(image, time_label,(100 , 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
      cv2.putText(image, fps_label,(100  , 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)    

      cv2.imshow('MediaPipe Face Mesh',image)
      if cv2.waitKey(1) == ord('q'):
        break

      out.write(image)  
  
  try:
    cap._stop()
    cap.release()
  except:
    print("Terminating")

  out.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  run()