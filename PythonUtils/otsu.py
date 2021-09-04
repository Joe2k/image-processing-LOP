from __main__ import app, s3
import flask 
import werkzeug
import time	
import cv2
import os
import time
import urllib
import numpy as np
import boto3, botocore
import io
import skimage.measure
import skimage.morphology
from matplotlib import pyplot as plt
from skimage.filters import sobel
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from PIL import Image

def Otsu():
    # print(flask.request.get_json()["url"])
    req = urllib.request.urlopen(flask.request.get_json()["url"])
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(arr, -1)
    # image = cv2.imread('./preProcessImages/test.png')
    scale_percent = 100 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_equalized = cv2.equalizeHist(gray)
    
    blur = cv2.GaussianBlur(image_equalized,(5,5),0)
    ret3,th3 = cv2.threshold(blur,150,230,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    def vdk_perimeter(th3):
        (w, h) = th3.shape
        data = np.zeros((w + 2, h + 2), dtype=th3.dtype)
        data[1:-1, 1:-1] = th3
        data = skimage.morphology.binary_dilation(data)
        newdata = np.copy(data)
        for i in range(1, w + 1):
            for j in range(1, h + 1):
                cond = data[i, j] == data[i, j + 1] and \
                    data[i, j] == data[i, j - 1] and \
                    data[i, j] == data[i + 1, j] and \
                    data[i, j] == data[i - 1, j]
                if cond:
                    newdata[i, j] = 0

        return np.count_nonzero(newdata)

    label_img = skimage.measure.label(th3)
    regions = skimage.measure.regionprops(label_img)

    image_string = cv2.imencode('.jpg', th3)[1].tobytes()
    imageName = flask.request.get_json()["filename"] + '_otsu.jpg'
    # cv2.imshow('Gray image', gray)
    area = 0 
    perimeter=0
    for props in regions:
        area= area+ props.area
        perimeter = perimeter + vdk_perimeter(props.convex_image)
    try:

        s3.upload_fileobj(
            io.BytesIO(image_string),
            os.environ.get("S3_BUCKET_NAME"),
            imageName,
            ExtraArgs={
                "ACL": "public-read",
                "ContentType": 'image/jpeg'
            }
        )

    except Exception as e:
        # This is a catch all exception, edit this part to fit your needs.
        print("Something Happened: ", e)
        return e
    print(os.environ.get("CLOUDFRONT_URL")+imageName)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Working")
    return {
        "url": os.environ.get("CLOUDFRONT_URL")+imageName,
        "perimeter": int(perimeter),
        "area":int(area)
    }

@app.route('/otsu1', methods=['GET', 'POST'])
def Otsu1():
    return Otsu()
@app.route('/otsu2', methods=['GET', 'POST'])
def Otsu2():
    return Otsu()
@app.route('/otsu3', methods=['GET', 'POST'])
def Otsu3():
    return Otsu()