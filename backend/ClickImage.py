#Face Registration
import cv2
import numpy as np
import os
from imageai.Detection import ObjectDetection

class ClickImage():
    
    
    def click_new_image(self):
        #Initialising Camera
        cap=cv2.VideoCapture(0) 
        detector = ObjectDetection()
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath("resnet50_coco_best_v2.1.0.h5")
        detector.loadModel()
        #detections = detector.detectObjectsFromImage(input_image="C:\\Users\\ekans\\Desktop\\Progress Report\\image.jpg")
        execution_path='./'
        
        while True:
            #reading image from webcam
            ret,frame=cap.read()
            if ret==False:
                continue
            
            cv2.imshow("Colored",frame)
            
            
            detections = detector.detectObjectsFromImage(input_image=frame, output_image_path=os.path.join(execution_path , "imagenew.jpg"))

            
            
            #for Clicking Images
            key_pressed=cv2.waitKey(1) & 0xff
            if key_pressed == ord('c'):
            #   cv2.imwrite('imagenew.jpg', frame)
              cv2.imshow('imagenew.jpg')

            #for quitting
            key_pressed=cv2.waitKey(1) & 0xff
            if key_pressed == ord('q'):
                break


        cap.release()
        cv2.destroyAllWindows()

        return True

image=ClickImage()
image.click_new_image()