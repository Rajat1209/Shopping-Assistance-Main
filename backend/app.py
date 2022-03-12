from flask import Flask, render_template,request
from flask.wrappers import Request
from imageai.Detection import ObjectDetection
# from gtts import gTTS
import os
# import vlc
# from translate import Translator

execution_path = os.getcwd()
language = 'en'
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.1.0.h5"))
detector.loadModel()
app= Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/prediction")
def prediction():
    
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))
    for Object in detections:
     a= Object["name"]
    #  myobj = gTTS(text=a, lang=language, slow=False)
    #  myobj.save("welcome.mp3")
    #  p= vlc.MediaPlayer("C:\\Users\\ekans\\Desktop\\MAJOR\\welcome.mp3")
    #  p.play()
    #  translator= Translator(to_lang="German")
    #  translation = translator.translate(a)
    #  amazon="https://www.amazon.in/s?k="+a 
    #  google="https://www.google.com/search?q="+a
    #  flipkart="https://www.flipkart.com/search?q="+a
     return {"a":a}

if __name__ == "__main__":
    app.run(debug=True)
