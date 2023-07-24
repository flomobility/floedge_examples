# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

import numpy as np
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import cv2
import time
from anx_interface import TfliteInterface, DeviceType, Anx

model_path = '/root/floedge_examples_dup/ai/homesecurity/human_detection/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/detect.tflite'
threshold = 0.5

anx = Anx()
anx.start_device_camera(fps=30, width=640, height=480, pixel_format=0)
video=anx.device_camera

class DetectorAPI:
    def __init__(self, model_path):
        self.tflite_interface = TfliteInterface(DeviceType.GPU)
        self.success = self.tflite_interface.load_model(model_path)
        self.image_tensor = self.tflite_interface.input_tensors
        self.output_tensor = self.tflite_interface.output_tensors
        
        print("Model loaded successfully" if self.success == True else "Error loading model")

        # Definite input and output Tensors for detection_graph
        # self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        # self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        # self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        # self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        # self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        #image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        #start_time = time.time()
        #(boxes, scores, classes, num) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],feed_dict={self.image_tensor: image_np_expanded})
        #end_time = time.time()

        #print("Elapsed Time:", end_time-start_time)

        height, width = self.image_tensor[0].shape[1], self.image_tensor[0].shape[2]
        resize_image = cv2.resize(image,(width, height))
        image_np_expanded = np.expand_dims(resize_image, axis=0)

        self.tflite_interface.set_inputs([image_np_expanded])

        start = time.time()
        self.tflite_interface.invoke()
        end = time.time()

        (boxes, classes, scores, num) = self.tflite_interface.get_output()
        
        im_height, im_width,_ = image.shape
        if boxes is None:
            raise Exception("Boxes is empty")
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))
        
        return boxes_list, classes[0].tolist(), scores[0].tolist(), int(num[0])

    def close(self):
        print("Unloading model")
        self.tflite_interface.unload_model()

def determine_if_person_in(fname,threshold=0.5,is_nano=True):
    print('Analyzing file {} to find human in video'.format(fname))
    odapi = DetectorAPI(model_path)
    jpeg_name=fname[:fname.rfind('.')+1]+'jpg'
    if not is_nano:
        return True,'There may be someone in the room. '
    video = cv2.VideoCapture(fname)    
    n_frames=video.get(cv2.CAP_PROP_FRAME_COUNT)

    max_budget=int(8*n_frames/100) if is_nano else int(4*n_frames/100)
    #print('max budget: {}'.format(max_budget))
    random_frames=np.random.permutation(int(n_frames))
    chosen_frames=random_frames[:max_budget]
    #print('random frames are: {}'.format(random_frames))
    #print('Chosen frames are: {}'.format(chosen_frames))
    

    for cf in chosen_frames:
        video.set(cv2.CAP_PROP_POS_FRAMES,cf)
        ret,frame=video.read()
        if not ret:
            video.release()
            #print('Could not read frame. Returning')
            message='Could not check the video.' 
            return True, message
        else:
            try:
                boxes, classes, scores, num= odapi.processFrame(frame)
            except Exception as e:
                   odapi.close()
                   video.release()
                   return False, str(e)
            if 0 in classes and scores[classes.index(0)]>threshold:
                for i in range(len(boxes)):
                    if classes[i] == 0 and scores[i] > threshold:
                        box=boxes[i]
                        cv2.rectangle(frame,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
                cv2.imwrite(jpeg_name,frame)
                message='There is someone in the room. '
                video.release()
                odapi.close()
                return True,message #if person is found in any frame, the function returns True
    message='Person not found in video. '
    video.release()
    odapi.close()
    return False,message


if __name__ == "__main__":
    flag, message = determine_if_person_in("/root/floedge_examples_dup/ai/homesecurity/raspi3/h264_videos/video3.mp4")
    print(message)
    #cap = cv2.VideoCapture('video.mp4')
    #count=0
    #while True:
        #r, img = video.read()
        #if not r:
            #continue
        #img = cv2.resize(img, (1280, 720))
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #count+=r

        #boxes, classes, scores, num= odapi.processFrame(img)
        # Visualization of the results of a detection.

        #for i in range(len(boxes)):
            # Class 1 represents human
            #print(classes[i], scores[i])
            #if classes[i] == 0 and scores[i] > threshold:
                #box = boxes[i]
                #cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
       
        #cv2.imshow("preview", img)
        #if count%50==0:
        #    cv2.imwrite('result{}.jpg'.format(count),img)

        #key = cv2.waitKey(1)
        #if key & 0xFF == ord('q'):
            #break
    #cv2.destroyAllWindows()