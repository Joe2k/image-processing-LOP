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

def vdk_perimeter(convert_matrix_copy1):
    (w, h) = convert_matrix_copy1.shape
    data = np.zeros((w + 2, h + 2), dtype=convert_matrix_copy1.dtype)
    data[1:-1, 1:-1] = convert_matrix_copy1
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

def SingleWatershed(thres_low, thres_high, num):
     # print(flask.request.get_json()["url"])
    req = urllib.request.urlopen(flask.request.get_json()["url"])
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(arr, 0)
    # image = cv2.imread('./preProcessImages/test.png')
     
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_equalized = cv2.equalizeHist(image)
    markers = np.zeros_like(image_equalized)
    markers[image_equalized < thres_low] = 1
    markers[image_equalized > thres_high] = 2
    elevation_map = sobel(image_equalized)

    markers = np.zeros_like(image_equalized)
    markers[image_equalized < thres_low] = 1
    markers[image_equalized > thres_high] = 2
    segmentation = watershed(elevation_map, markers)
    segmentation = ndi.binary_fill_holes(segmentation - 1)

    im1 = np.array(segmentation * 255, dtype = np.uint8)
    image = Image.fromarray(im1)


    label_img = skimage.measure.label(segmentation)
    regions = skimage.measure.regionprops(label_img)

    area_2=0
    peri_2=0
    for props in regions:
        area_2+=props.area
        peri_2+=vdk_perimeter(props.convex_image)
        #print (props.area, vdk_perimeter(props.convex_image))
   
    # segmentation=np.array(segmentation)
    # cv2.imshow('Original image',img)
    # cv2.imshow('Canny image', edges)
    # cv2.imwrite("./postProcessImages/Canny.jpg", edges)
    image_string = cv2.imencode('.jpg', np.array(image))[1].tobytes()
    imageName = flask.request.get_json()["filename"] + '_singlewatershed' +  str(num) + '.jpg'
    # cv2.imshow('Gray image', gray)
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
        "url": os.environ.get("CLOUDFRONT_URL") + imageName,
        "perimeter": int(peri_2),
        "area":int(area_2)
    }

# SingleWatershed Method
@app.route('/singlewatershed1', methods=['GET', 'POST'])
def SingleWatershed1():
    
    return SingleWatershed(150, 230, 1)

@app.route('/singlewatershed2', methods=['GET', 'POST'])
def SingleWatershed2():
    
    return SingleWatershed(20, 230, 2)

@app.route('/singlewatershed3', methods=['GET', 'POST'])
def SingleWatershed3():
    
    return SingleWatershed(20, 250, 3)