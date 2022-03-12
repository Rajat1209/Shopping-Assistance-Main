# import ClickImage
import cv2
import os
from imageai.Detection import ObjectDetection


class DetectObject():

    def detect(self):
        # clickNewImage=ClickImage()
        # clickNewImage.click_new_image()

        
        
        
        detector = ObjectDetection()
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath("resnet50_coco_best_v2.1.0.h5")
        detector.loadModel()
        #detections = detector.detectObjectsFromImage(input_image="C:\\Users\\ekans\\Desktop\\Progress Report\\image.jpg")
        execution_path='./'
        detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))

            
        for eachObject in detections:
            print(eachObject["name"] , " : " , eachObject["percentage_probability"] )


obj=DetectObject()
obj.detect()
