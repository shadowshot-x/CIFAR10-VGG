from flask import Flask, render_template, request,redirect
from io import BytesIO
from keras.preprocessing import image
from keras.preprocessing.image import array_to_img, img_to_array,load_img
from keras.models import load_model,model_from_json
import os
from PIL import Image
import tensorflow as tf
from flask_wtf import FlaskForm
from flask_wtf.file import FileField,FileRequired,FileAllowed
from wtforms import SubmitField
from werkzeug.utils import secure_filename
from flask_bootstrap import Bootstrap

label_dict = {0:'Airplane', 1:'Automobile', 2:'Bird', 3:'Cat', 4:'Deer', 5:'Dog',6:'Frog', 7:'Horse', 8:'Ship', 9:'Truck'}

app = Flask(__name__)

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY
Bootstrap(app)

class PhotoForm(FlaskForm):
    photo = FileField(validators=[FileRequired(),FileAllowed(['jpg','jpeg', 'png'], 'Images only!')])
    submit = SubmitField('Submit')

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
@app.route("/",methods=["GET","POST"])
def home():
    return "Hello, World!"

@app.route("/predict",methods=["GET","POST"])
def predict():
    img=load_img("./static/download.jpeg",target_size=(32, 32))
    img = img_to_array(img)
    img = img.reshape(1, 32, 32, 3)
    img = img.astype('float32')
    img=img/255.0
    result = model.predict_classes(img)
    response="ML Predictions"
    return "yes"

@app.route("/upload",methods=["GET","POST"])
def upload():
    form=PhotoForm()
    if form.validate_on_submit():
        image_stream=form.photo.data.stream
        original_img=Image.open(image_stream)
        original_img=original_img.resize((32,32))
        img = img_to_array(original_img)
        img = img.reshape(1, 32, 32, 3)
        img = img.astype('float32')
        img=img/255.0
        result = model.predict_classes(img)
        ans=label_dict[result[0]]
        return render_template("index.html",result=ans)
    return render_template('index.html', form=form)


    
if __name__ == "__main__":
    app.run(debug=True)