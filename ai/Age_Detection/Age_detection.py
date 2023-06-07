from anx_interface import TfliteInterface, DeviceType
import numpy as np
import cv2
from time import time

string_pred_age = ['04 - 06 ani', '07 - 08 ani','09 - 11 ani','12 - 19 ani','20 - 27 ani','28 - 35 ani','36 - 45 ani','46 - 60 ani','61 - 75 ani']

# Load TFLite model and allocate tensors. Load Face Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

tflite_interface_age = TfliteInterface(DeviceType.GPU) 
tflite_interface_age.load_model("AgeClass_best_06_02-16-02.tflite")

input_im = None

webcam = cv2.VideoCapture("/root/Downloads/UPSC_Sample.mp4")

webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    font = cv2.FONT_HERSHEY_SIMPLEX
    time_start = time()
    success, frame = webcam.read()
    size = (640,480)
    result = cv2.VideoWriter('Age_result.mp4', 
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             12, size)  
    start = time()
    end = time()  
    if not success:
        continue
        
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor = 1.2, minNeighbors=5)
    prev_fps = ""
    for x,y,w,h in faces:
        saved_image = frame          
        input_im = saved_image[y:y+h, x:x+w]
        
        if input_im is None:
            print("No Face")
        else:
            input_im = cv2.resize(input_im, (224,224))
            input_im = input_im.astype('float')
            input_im = input_im / 255
            input_im = np.array(input_im)
            input_im = np.expand_dims(input_im, axis = 0)

            # Predict
            input_data = np.array(input_im, dtype=np.float32)
    
            tflite_interface_age.set_inputs([input_data])
            start = time()
            tflite_interface_age.invoke()
            end = time()
            
            print((end-start)*1000)
            output_data_age = tflite_interface_age.get_output()
            index_pred_age = int(np.argmax(output_data_age))
            prezic_age = string_pred_age[index_pred_age]
            
            cv2.putText(frame,prezic_age, (x,y), font, 1, (255,255,255), 2, cv2.LINE_AA)
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
        tflite_interface_age.unload_model()
        break
