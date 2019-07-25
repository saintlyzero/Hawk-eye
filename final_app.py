
from flask import Flask
from flask import Flask,render_template,url_for,request
from flask_cors import CORS, cross_origin
from flask_pymongo import PyMongo
import json, jsonify
from werkzeug import secure_filename
from bson.json_util import dumps
from bson.objectid import ObjectId

import os
import re
import argparse
import datetime
import pprint
from google.cloud import storage


UPLOAD_FOLDER = '/home/naresh/flask_app/uploads'
bucket_name = 'b-ao-locale-19'
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
app.config["MONGO_URI"] = "mongodb://127.0.0.1:27017/athenas-owl"
mongo = PyMongo(app)
CORS(app)


@app.route('/')
def hello():
    # return "Hello World!"
    online_users = mongo.db.videos.find({})
    res = dumps(online_users)
    print("\n\n\n\n",res, "\n\n\n")
    return res
    

@app.route('/get_user_videos/<id>')
def get_user_videos(id):
    return dumps(mongo.db.videos.find( { "user_id": id }, { "video_url": 1, "status": 1, "video_name" : 1  }).sort("_id", -1))


@app.route('/get_results/<id>')
def get_results(id):
    return dumps(mongo.db.videos.find( {"_id": ObjectId(id)}, { "locale_response" : 1, "angle_response" : 1, "apparel_response" : 1}))


@app.route('/get_video_status/', methods = ['GET'] )
def get_video_status():
    video_url = request.args.get('url')
    if video_url:        
        return dumps(mongo.db.videos.find_one( { "video_url": video_url}, { "status": 1, "apparel_response" : 1, "locale_response" : 1, "angle_response" : 1}))
    else:
        return "key error occured, use proper key and ID"
        

@app.route('/fetch_by_oid/<id>')
def fetch_by_oid(id):
    return dumps(mongo.db.videos.find( { "_id": ObjectId(id) }, { "video_url": 1, "status": 1, "_id":0}))


@app.route('/upload_video', methods=['POST'])
def upload_blob():
    if request.method =='POST':
        file = request.files['file']
        user_id = request.form.get('userid')

        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))

            source_file_name = os.path.join(UPLOAD_FOLDER,filename)
            destination_blob_name = os.path.join('test',file.filename)
            """Uploads a file to the bucket."""
            storage_client = storage.Client()
            bucket = storage_client.get_bucket(bucket_name)
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(source_file_name)
            print('File {} uploaded to {}.'.format(
                source_file_name,
                destination_blob_name))

            gcs_url = 'https://storage.cloud.google.com/%(bucket)s/%(file)s' % {'bucket':bucket_name, 'file':destination_blob_name}
            db_entry = {
             "video_url": gcs_url,
             "video_name": filename,
             "local_path": os.path.join(UPLOAD_FOLDER,filename),
             "user_id": user_id,
             "status": "pending",
             "apparel_response": None,
             "locale_response": None,
             "angle_response": None,
             }
            mongo.db.videos.insert(db_entry)
            # return dumps(db_entry)
            return dumps({"gcs_url": gcs_url, "success": "true"})
        return "file uploading error occurred"
    return "use proper method"


if __name__ == '__main__':
    app.run(port=8081, debug= True, host='0.0.0.0', threaded=True)



# db.videos.insert({
#     user_id: "102",
#     path: "uploads/js.mp4",
#     status: "pending",
#     locale: null,
#     angle: null,
#     apparel:null });

# @app.route('/check', methods = ['GET'])
# def get_video():
#     user_id = request.args.get('id')
#     if user_id:
#         return dumps(mongo.db.videos.find( { "user_id": user_id }, { "video_url": 1, "status": 1, "_id": 0 }))
#     else:
#         return "key error occured, use proper key and ID"
