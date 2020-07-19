from flask import Flask, render_template, request
from io import BytesIO
from keras.preprocessing import image
from keras.preprocessing.image import array_to_img, img_to_array,load_img
from keras.models import load_model,model_from_json
import os
from PIL import Image
import tensorflow as tf

label_dict = {0:'Cat', 1:'Giraffe', 2:'Sheep', 3:'Bat', 4:'Octopus', 5:'Camel',6:'Cat', 7:'Giraffe', 8:'Sheep', 9:'Bat'}

app = Flask(__name__)

def init():
    json_file=open("model/model.json","r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model/model.h5")
    print("Loaded Model from disk")
    loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return loaded_model

model = init()
@app.route("/")
def home():
    return "Hello, World!"

@app.route("/predict",methods=["GET","POST"])
def predict():
    img=load_img("./static/hybrid-car-ch.jpg",target_size=(32, 32))
    img = img_to_array(img)
    img = img.reshape(1, 32, 32, 3)
    img = img.astype('float32')
    img=img/255.0
    print("reached")
    result = model.predict_classes(img)
    print(result)
    response="ML Predictions"
    return "yes"

    
if __name__ == "__main__":
    app.run(debug=True)