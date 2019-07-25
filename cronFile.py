import pymongo
import json
from pprint import pprint
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
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import time
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from flask import Flask
from flask import Flask,render_template,url_for,request
from flask_cors import CORS, cross_origin
import pandas as pd 
from werkzeug import secure_filename
from bson.json_util import dumps
import os

def load_trained_locale_model():
    img_height,img_width = 256,256
    num_classes = 16
    # load pretrained model
    base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (img_height,img_width,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    #x = Dropout(0.7)(x)
    predictions = Dense(num_classes, activation= 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    model=load_model('/home/naresh/flask_app/model/locale_model/weights-improvement-03-0.90.hdf5')
    return model


def predict_locale(test_img,j, model):
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
        predictions = [ [labels[k],str(round(conf_val,3))] for k,conf_val in zip(predicted_class_indices,conf_score)]
        return predictions
    


def predict_from_locale_model(pathIn, model):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line 
        success,image = vidcap.read()
        output.append(predict_locale(image,count,model))
        count = count + 1
                
        
def load_trained_angle_model():
    img_height,img_width = 256,256
    num_classes = 16
    # load pretrained model
    base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (img_height,img_width,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    #x = Dropout(0.7)(x)
    predictions = Dense(num_classes, activation= 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    model=load_model('/home/naresh/flask_app/model/angle_model/weights.01-0.61.hdf5')
    return model

def predict_angle(test_img,j, model):
    if test_img is not None:
        id_val.append(j)
        test_img=cv2.resize(test_img,(256,256))
        test_img=np.reshape(test_img,(1,256,256,3))
        test_img=test_img/255.
        classes = model.predict(test_img)
        conf_score=np.max(classes,axis=1)
        predicted_class_indices=np.argmax(classes,axis=1)
        labels = {'bird_eye' : 0,'close_up':1,'high_angle':2,'long_shot':3,'low_angle':4,'medium_shot':5,'other':6}
        labels = dict((v,k) for k,v in labels.items())
        predictions = [[labels[k],str(round(conf_val,3))] for k,conf_val in zip(predicted_class_indices,conf_score)]
        return predictions        
        
def predict_from_angle_model(pathIn, model):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line 
        success,image = vidcap.read()
        #print(image.shape)
        output.append(predict_angle(image,count,model))
        count = count + 1


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def get_frame_pred(image_path, model):
    frame_pred = []
    image = read_image_bgr(image_path)
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    image = preprocess_image(image)
    image, scale = resize_image(image)
    #print(image)
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    # correct for image scale
    boxes /= scale
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.2:
            break
        color = label_color(label)
        b = box.astype(int)
        draw_box(draw, b, color=color)
        frame_pred.append((b , labels_to_names[label], score))
        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
    frames_caption = '/home/naresh/flask_app/model/apparel_model/frames_caption'
    if os.path.exists(frames_caption)==False:
        os.mkdir(frames_caption)
    cv2.imwrite(frames_caption + '/' + image_path.split('/')[-1], draw)
    return frame_pred, draw.shape[0], draw.shape[1]


def extract_images(video_path, path_out):
    frames = []
    count = 0
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line 
        success,image = vidcap.read()
#        print('Read a new frame: ', success)
        if success == True:
            path = path_out + '/' + "frame_%d.jpg" % count
            cv2.imwrite(path, image)     # save frame as JPEG file
            frames.append(path)        
        count = count + 1
    return frames

def run_inference(video_path, model_path):
    # load retinanet model
    model = models.load_model(model_path, backbone_name='resnet50')
    frames_path = '/home/naresh/flask_app/model/apparel_model/video_frames'
    if os.path.exists(frames_path)==False:
        os.mkdir(frames_path)
    frames = extract_images(video_path, frames_path)
    prediction = []
    frame_id = 0
    for frame_path in frames:
        frame_pred, height, width = get_frame_pred(frame_path,model)
        outcomes = []
        for pred in frame_pred:
            bbox, label, confidence = pred
            x_min, y_min, x_max, y_max = bbox#
            outcomes.append([ str(label),  str(round(confidence,2)) ])
        if outcomes == []:
            frame_json = {str(frame_id): [outcomes]}
        else:
            frame_json = {str(frame_id): outcomes}        
        prediction.append(frame_json)
        frame_id = frame_id + 1    
    result_json = prediction
    return result_json
        
    

myclient = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
mydb = myclient["athenas-owl"]
mycol = mydb["videos"]


pathIn = ''
pending_file = mycol.find_one({'status': {"$in" : ['pending'] } })

if pending_file != None:
    mycol.update_one(pending_file, { "$set": { "status": "processing" } })
    file_id = pending_file["_id"]
    pathIn =  pending_file["local_path"]
    print("file's object id is: ", file_id)  

    id_val = []
    output = []            
    print("loading Locale model")
    locale_model =load_trained_locale_model()
    print("Locale model loaded") 
    predict_from_locale_model(pathIn, locale_model)
    locale_result = dict(zip(id_val,output))
    print("result is produced for locale model")

    id_val = []
    output = []        
    print("loading Angle model")
    angle_model =load_trained_angle_model()
    print("Angle model loaded") 
    predict_from_angle_model(pathIn, angle_model)
    angle_result = dict(zip(id_val,output))
    print("result is produced for angle model")

    print("making predictions Apparel model")
    df = pd.read_csv('/home/naresh/flask_app/model/apparel_model/classes.csv', header=None)
    labels_to_names = {}
    for label, index in zip(df[0], df[1]):
        labels_to_names[index]=label
 
    weights = '/home/naresh/flask_app/model/apparel_model/model.h5'
    result_json = run_inference(pathIn, weights)    
    output_json=json.dumps(result_json)
    print(type(output_json))
    output_json = output_json.replace('{', '')
    output_json = output_json.replace('}', '')    
    l = list(output_json)
    l[0] = '{'
    l[-1] = '}'
    apparel_result = ''.join(l)

    mycol.update_one({"_id": file_id}, { "$set": { "locale_response": dumps(locale_result), "angle_response": dumps(angle_result), "apparel_response": apparel_result ,"status": "completed" } })

    print("file is processed having file_id", str(file_id))

else:
    print("nothing to process")
    exit(0)


#for x in mycol.find():
#    pprint(x) 
# model_result1 = model1.run(pending_file.path)
# model_result2 = model2.run(pending_file.video_url)
# model_result3 = model3.run(pending_file.video_url)
# mongo.update({'_id' : pending_file._id}, {
# "status": "complete",
# "apparel_response": model_result1,
# "locale_response": model_result2,
# "angle_response": model_result3,
# })
# print("File completed")