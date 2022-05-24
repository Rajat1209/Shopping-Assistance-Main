from flask import Flask, render_template,request
from flask.wrappers import Request
from imageai.Detection import ObjectDetection
import os
import pyrebase
import base64

firebaseConfig= {
  "apiKey": "AIzaSyAZTu8rdlqbSh5Q8ENfcufwziSfYLtnYYU",
  "authDomain": "shopping-assitance.firebaseapp.com",
  "databaseURL": "https://shopping-assistance-final-viva-default-rtdb.firebaseio.com",
  "projectId": "shopping-assitance",
  "storageBucket": "shopping-assitance.appspot.com",
  "messagingSenderId": "104214180012",
  "appId": "1:104214180012:web:139e8bc3eb046d513aaf09"
};

execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.1.0.h5"))
detector.loadModel()
app= Flask(__name__)

firebase=pyrebase.initialize_app(firebaseConfig)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/prediction")
def prediction():
    db=firebase.database()
    image=db.child("Image").get()
    #a=image.val()
    data=''
    for image in image.each():
        data=image.val()['imageData']

    with open("image.jpg", "wb") as fh:
        fh.write(base64.urlsafe_b64decode(data))
    
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))
    db.child("Image").remove() 

    object_found="No Object Detected!"
    object_probability=0
    for eachObject in detections:
        if(object_probability < eachObject["percentage_probability"]):
            object_found=eachObject["name"]
            object_probability=eachObject["percentage_probability"]



    return {"object":object_found}

if __name__ == "__main__":
    app.run(debug=True)
