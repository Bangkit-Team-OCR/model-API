from flask import Flask, request, jsonify
import cv2
import tensorflow as tf
import numpy as np
import os
import shutil
import string
import pytesseract
import urllib.request as url

model = tf.keras.models.load_model('my_model.h5')

def processImage(img):
    req = url.urlopen(img)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, 0)
    # img=cv2.imread(img,0)
    ret,img=cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU,img)
    img=cv2.resize(img,(512,512))
    img= np.expand_dims(img,axis=-1)
    img=img/255
    img=np.expand_dims(img,axis=0)
    return img
    
def createDir(file_path,dir):
  dir_path=os.path.join(file_path+dir+'/')
  if not os.path.exists(os.path.join(file_path,dir)):
    os.mkdir(file_path+dir)
  return dir_path


# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def hello():
    if request.method == 'POST':
        uuid = request.json['uuid']
        dir='dataCrop'
        file_path = f'{uuid}/' 
        createDir('./', file_path)
        dir_path = createDir(file_path,dir)
        print(dir_path)
        imgpath = request.json['imgpath']
        img=processImage(imgpath)
        pred=model.predict(img)
        pred=np.squeeze(np.squeeze(pred,axis=0),axis=-1)
        # # upload to bucket
        
        cv2.imwrite(file_path+'/mask.JPG',pred) 

        img = cv2.imread(file_path+'/mask.JPG',0)
        cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU,img)
        req = url.urlopen(imgpath)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        ori_img = cv2.imdecode(arr, 0)
        # ori_img=cv2.imread('/content/banten-0.png')
        ori_img=cv2.resize(ori_img,(512,512))
        contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        i=0
        for c in contours:
            # get the bounding rect
            x, y, w, h = cv2.boundingRect(c)
            # draw a white rectangle to visualize the bounding rect
            cv2.rectangle(ori_img, (x, y), (x+w,y+h), 255, 1)
            # coordinates.append((x,y,(x+w),(y+h)))
            if(w>15 and h>15) :
                print(i)
                new=ori_img[y:y+h,x:x+w]
                cv2.imwrite(f'{dir_path}/{i}.png', new)
                i+=1
        list_crop=os.listdir(dir_path)
        filename_list=[filename.split(".")[0]for filename in list_crop]
        text_list=[]
        filename_list.sort()
        print(filename_list)
        i=0
        for filename in filename_list:
            name=f"{dir_path}/{i}.png"
            t1 = pytesseract.image_to_string(name,lang='ind',config='--psm 10')
            text_list.append(t1.title())
            i+=1
        
        shutil.rmtree(file_path)
        return jsonify(text_list)
    
    return "method not allowed"


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. You
    # can configure startup instructions by adding `entrypoint` to app.yaml.
    app.run(host='0.0.0.0', debug=True)
