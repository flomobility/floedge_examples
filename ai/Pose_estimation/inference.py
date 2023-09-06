#!/usr/bin/env python3

from PIL import Image, ImageDraw 
import cv2
from anx_interface import TfliteInterface, DeviceType, Anx
import sys
import click
import numpy as np

MIN_CONFIDENCE = 0.60
@click.command()
@click.option("--video-src", help="Specifiy video source (device_camera/usb_camera/video).", required = False, default = "device_camera")
@click.option("--dev", help="USB camera.", required = False, default = "/dev/video3")
@click.option("--video-path", help="Path to video file.", required = False, default = "yoga.mp4")
@click.option("--model", help="Path to model file.", required = False, default = "./posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite")
@click.option("--width", help="Width of frame to capture from camera.", required = False, default = 640)
@click.option("--height", help="Height of frame to capture from camera.", required = False, default = 480)
def run(video_src, dev, video_path, model, width, height) -> None:
	import posenet
	body_joints = [[posenet.BodyPart.LEFT_WRIST, posenet.BodyPart.LEFT_ELBOW],
	               [posenet.BodyPart.LEFT_ELBOW, posenet.BodyPart.LEFT_SHOULDER],
	               [posenet.BodyPart.LEFT_SHOULDER, posenet.BodyPart.RIGHT_SHOULDER],
	               [posenet.BodyPart.RIGHT_SHOULDER, posenet.BodyPart.RIGHT_ELBOW],
	               [posenet.BodyPart.RIGHT_ELBOW, posenet.BodyPart.RIGHT_WRIST],
	               [posenet.BodyPart.LEFT_SHOULDER, posenet.BodyPart.LEFT_HIP],
	               [posenet.BodyPart.LEFT_HIP, posenet.BodyPart.RIGHT_HIP],
	               [posenet.BodyPart.RIGHT_HIP, posenet.BodyPart.RIGHT_SHOULDER],
	               [posenet.BodyPart.LEFT_HIP, posenet.BodyPart.LEFT_KNEE],
	               [posenet.BodyPart.LEFT_KNEE, posenet.BodyPart.LEFT_ANKLE],
	               [posenet.BodyPart.RIGHT_HIP, posenet.BodyPart.RIGHT_KNEE],
	               [posenet.BodyPart.RIGHT_KNEE, posenet.BodyPart.RIGHT_ANKLE]]

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

	posenet = posenet.PoseNet(model_path=model)

	while True:
		success, image = cap.read()
		if not success:
			continue
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if video_src != "device_camera" else image
		person = posenet.estimate_pose(image)
		image = Image.fromarray(image)
		draw = ImageDraw.Draw(image)
		for line in body_joints:

			if person.keyPoints[line[0].value[0]].score > MIN_CONFIDENCE and person.keyPoints[line[1].value[0]].score > MIN_CONFIDENCE:
				start_point_x, start_point_y = int(person.keyPoints[line[0].value[0]].position.x), int(person.keyPoints[line[0].value[0]].position.y)
				end_point_x, end_point_y = int(person.keyPoints[line[1].value[0]].position.x), int(person.keyPoints[line[1].value[0]].position.y)
				draw.line((start_point_x, start_point_y, end_point_x, end_point_y), fill=(255, 255, 255), width=2)

		for key_point in person.keyPoints:
			if key_point.score > MIN_CONFIDENCE:
				left_top_x, left_top_y = int(key_point.position.x) - 4, int(key_point.position.y) - 4
				right_bottom_x, right_bottom_y = int(key_point.position.x) + 4, int(key_point.position.y) + 4
				#center = (int((left_top_x+right_bottom_x)/2), int((left_top_y+right_bottom_y)/2))
				#axes = (right_bottom_x - left_top_x, right_bottom_y - left_top_y)
				draw.ellipse((left_top_x, left_top_y, right_bottom_x, right_bottom_y), fill=(255, 255, 255), outline=1)
		image = np.array(image)
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		print('total score : ', person.score)
		cv2.imshow("Output", image)
		out.write(image)
		if cv2.waitKey(1) == ord('q'):
			break  
	try:
		cap._stop()
		cap.release()
	except:
		print("Terminating")
 
	out.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	run()