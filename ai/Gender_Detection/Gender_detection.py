from anx_interface import TfliteInterface, DeviceType
import numpy as np
import cv2
from time import time

string_pred_age = ['04 - 06 ani', '07 - 08 ani','09 - 11 ani','12 - 19 ani','20 - 27 ani','28 - 35 ani','36 - 45 ani','46 - 60 ani','61 - 75 ani']
string_pred_gen = ['Female', 'Male']

# Load TFLite model and allocate tensors. Load Face Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

tflite_interface_gender = TfliteInterface(DeviceType.GPU) 
tflite_interface_gender.load_model("GenderClass_06_03-20-08.tflite")

input_im = None

webcam = cv2.VideoCapture("/root/Downloads/UPSC_Sample.mp4")

webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    font = cv2.FONT_HERSHEY_SIMPLEX
    time_start = time()
    success, frame = webcam.read()
    size = (1280,720)
    result = cv2.VideoWriter('Gender_result.mp4', 
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             12, size)
    
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
    cv2.imshow("Detecting faces...", frame)
    result.write(frame)
    
    key = cv2.waitKey(1)

    if key == ord('q') or key == ord('Q'):
        webcam.release()
        cv2.destroyAllWindows()
        result.release()
        tflite_interface_gender.unload_model()
        break
