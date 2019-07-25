import numpy as np
import cv2
import os
import pandas as pd
import tensorflow as tf
from PIL import Image
from keras.preprocessing import image
from keras import optimizers
from keras import applications
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
from sklearn.utils import shuffle
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from flask import Flask
from flask import Flask,render_template,url_for,request
from flask_cors import CORS, cross_origin
import pandas as pd 
from werkzeug import secure_filename
import pickle
import os
import json

UPLOAD_FOLDER = './uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
CORS(app)

@app.route('/')
def hello():
    return "Hello World!"

@app.route('/<name>')
def hello_name(name):
    return "Hello {}!".format(name)


@app.route('/upload',methods=['POST'])
def upload():
    if request.method =='POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            return "{\"success\":\"true\"}"
        return "file uploading error occurred"
    return "use proper method"

# from flask import jsonify
# from flask import json

# @app.route('/summary')
# def summary():
#     data = make_summary()
#     response = app.response_class(
#         response=json.dumps(data),
#         status=200,
#         mimetype='application/json'
#     )
#     return response

def load_trained_model():
    img_height,img_width = 256,256
    num_classes = 16
    # load pretrained model
    base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (img_height,img_width,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    #x = Dropout(0.7)(x)
    predictions = Dense(num_classes, activation= 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    model=load_model('weights-improvement-17-0.67.hdf5')
    return model

id_val=[]
def predict(test_img,j):
    if test_img is not None:
        id_val.append(j)
        test_img=cv2.resize(test_img,(256,256))
        test_img=np.reshape(test_img,(1,256,256,3))
        test_img=test_img/255.
        classes = model.predict(test_img)
        conf_score=np.max(classes,axis=1)
        predicted_class_indices=np.argmax(classes,axis=1)
        labels = {'bedroom': 0,'building': 1,'classroom': 2,'conference_room': 3,'corridor': 4,'dining_room': 5,'eatery': 6,
         'ground': 7,'hall': 8,'kitchen': 9,'living_room': 10,'mountain': 11,'office': 12,'railway_station': 13,'swimming_pool': 14,
         'waterbody': 15}
        labels = dict((v,k) for k,v in labels.items())
        predictions = [[labels[k],str(round(conf_val,3))] for k,conf_val in zip(predicted_class_indices,conf_score)]
        return predictions

def predict_from_model(pathIn):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line 
        success,image = vidcap.read()
        output.append(predict(image,count))
        count = count + 1
        

if __name__ == '__main__':
    app.run(port=5000, debug= True, host='0.0.0.0', threaded=True)