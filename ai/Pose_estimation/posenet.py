import math
import time
from enum import Enum
from anx_interface import TfliteInterface, DeviceType, Anx
import numpy as np
from PIL import Image
import cv2

class BodyPart(Enum):
	NOSE = 0,
	LEFT_EYE = 1,
	RIGHT_EYE = 2,
	LEFT_EAR = 3,
	RIGHT_EAR = 4,
	LEFT_SHOULDER = 5,
	RIGHT_SHOULDER = 6,
	LEFT_ELBOW = 7,
	RIGHT_ELBOW = 8,
	LEFT_WRIST = 9,
	RIGHT_WRIST = 10,
	LEFT_HIP = 11,
	RIGHT_HIP = 12,
	LEFT_KNEE = 13,
	RIGHT_KNEE = 14,
	LEFT_ANKLE = 15,
	RIGHT_ANKLE = 16,


class Position:
	def __init__(self):
		self.x = 0
		self.y = 0


class KeyPoint:
	def __init__(self):
		self.bodyPart = BodyPart.NOSE
		self.position = Position()
		self.score = 0.0


class Person:
	def __init__(self):
		self.keyPoints = []
		self.score = 0.0


class PoseNet:
	def __init__(self, model_path):
		self.input_mean = 127.5
		self.input_std = 127.5
		self.image_width = 0
		self.image_height = 0
		self.tflite_interface = TfliteInterface(DeviceType.GPU)
		self.success = self.tflite_interface.load_model(model_path)
		self.input_details = self.tflite_interface.input_tensors
		self.output_details = self.tflite_interface.output_tensors
		print("Model loaded successfully" if self.success == True else "Error loading model")

	def sigmoid(self, x):
		return 1. / (1. + math.exp(-x))

	def load_input_image(self, image):
		height, width = self.input_details[0].shape[1], self.input_details[0].shape[2]
		input_image = Image.fromarray(image)
		self.image_width, self.image_height = input_image.size
		resize_image = input_image.resize((width, height))
		return np.expand_dims(resize_image, axis=0)

	def estimate_pose(self, image):
 		
		input_data = self.load_input_image(image)

		if self.input_details[0].dtype == type(np.float32(1.0)):
			input_data = (np.float32(input_data) - self.input_mean) / self.input_std

		self.tflite_interface.set_inputs([input_data])

		start = time.time()

		self.tflite_interface.invoke()

		end = time.time()
		dt = (end-start)*1000
		print("dt: ",dt)
		output_data = self.tflite_interface.get_output()
		heat_maps = np.copy(output_data[0])
		offset_maps = np.copy(output_data[1])

		height = len(heat_maps[0])
		width = len(heat_maps[0][0])
		num_key_points = len(heat_maps[0][0][0])


		key_point_positions = [[0] * 2 for i in range(num_key_points)]
		for key_point in range(num_key_points):
			max_val = heat_maps[0][0][0][key_point]
			max_row = 0
			max_col = 0
			for row in range(height):
				for col in range(width):
					heat_maps[0][row][col][key_point] = self.sigmoid(heat_maps[0][row][col][key_point])
					if heat_maps[0][row][col][key_point] > max_val:
						max_val = heat_maps[0][row][col][key_point]
						max_row = row
						max_col = col
			key_point_positions[key_point] = [max_row, max_col]

		x_coords = [0] * num_key_points
		y_coords = [0] * num_key_points
		confidenceScores = [0] * num_key_points
		for i, position in enumerate(key_point_positions):
			position_y = int(key_point_positions[i][0])
			position_x = int(key_point_positions[i][1])
			y_coords[i] = (position[0] / float(height - 1) * self.image_height +
			               offset_maps[0][position_y][position_x][i])
			x_coords[i] = (position[1] / float(width - 1) * self.image_width +
			               offset_maps[0][position_y][position_x][i + num_key_points])
			confidenceScores[i] = heat_maps[0][position_y][position_x][i]

		person = Person()
		key_point_list = []
		for i in range(num_key_points):
			key_point = KeyPoint()
			key_point_list.append(key_point)
		total_score = 0
		for i, body_part in enumerate(BodyPart):
			key_point_list[i].bodyPart = body_part
			key_point_list[i].position.x = x_coords[i]
			key_point_list[i].position.y = y_coords[i]
			key_point_list[i].score = confidenceScores[i]
			total_score += confidenceScores[i]

		person.keyPoints = key_point_list
		person.score = total_score / num_key_points

		return person