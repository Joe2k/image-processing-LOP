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

class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x
    def getY(self):
        return self.y

def getGrayDiff(img,currentPoint,tmpPoint):
    return abs(int(img[currentPoint.x,currentPoint.y]) - int(img[tmpPoint.x,tmpPoint.y]))

def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1),  Point(0, 1), Point(-1, 1), Point(-1, 0)]
    else:
        connects = [ Point(0, -1),  Point(1, 0),Point(0, 1), Point(-1, 0)]
    return connects

def regionGrow(img,seeds,thresh,p = 1):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []
    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects(p)
    while(len(seedList)>0):
        
        currentPoint = seedList.pop(0)

        seedMark[currentPoint.x,currentPoint.y] = label
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))
            if grayDiff < thresh and seedMark[tmpX,tmpY] == 0:
                seedMark[tmpX,tmpY] = label
                seedList.append(Point(tmpX,tmpY))
    return seedMark

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

def SingleRegion(threshold, num):
    # print(flask.request.get_json()["url"])
    req = urllib.request.urlopen(flask.request.get_json()["url"])
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(arr, 0)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_eq = cv2.equalizeHist(image)
    height1, weight1 = img_eq.shape
    h1 = int(height1/2)
    w1 = int(weight1/2)
    #seeds = (h1, w1)
   

    seed1 = [Point(h1, w1), Point(h1+2, w1+2), Point(h1+3, w1+3), Point(h1+4, w1+4), Point(h1+5, w1+5), Point(h1+6, w1+6), Point(h1+7, w1+7), Point(h1+8, w1+8)]
    seed2 = [Point(h1-24, w1-24), Point(h1-20, w1-20), Point(h1-36, w1-36), Point(h1-14, w1-14), Point(h1-10, w1-10), Point(h1-6, w1-6), Point(h1-7, w1-7), Point(h1-8, w1-8), Point(h1-5, w1-5)]
    seed3 = [Point(h1, -w1), Point(h1+5, w1-5), Point(h1+5, w1-8), Point(h1+2, w1-8), Point(h1+2, w1+8)]
    seed4 = [Point(h1-5, w1+5), Point(h1-5, w1+8), Point(h1+5, w1-5), Point(h1+5, w1-8),Point(h1-5, w1-2), Point(h1-5, w1-5), Point(h1-5, w1-8) ]
    seed5 = [Point(h1-14, w1+5), Point(h1-14, w1+2), Point(h1-14, w1+1), Point(h1-14, w1+3), Point(h1-14, w1+4), Point(h1-10, w1+5), Point(h1-10, w1+2), Point(h1-10, w1+1), Point(h1-10, w1+3), Point(h1-10, w1+4)]
    seed6 = [Point(h1+24, w1+24), Point(h1+20, w1+20), Point(h1+36, w1+36)]

    seed7 = [Point(h1-15, w1+15), Point(h1-15, w1+18), Point(h1+15, w1-15), Point(h1+15, w1-18),Point(h1-15, w1-12), Point(h1-15, w1-15), Point(h1-15, w1-18) ]
    seed8 = [Point(h1-24, w1+15), Point(h1-24, w1+12), Point(h1-24, w1+10), Point(h1-24, w1+13), Point(h1-24, w1+14), Point(h1-20, w1+15), Point(h1-20, w1+12), Point(h1-20, w1+10), Point(h1-20, w1+13), Point(h1-20, w1+14)]
    


    seeds = seed1  + seed2 + seed3 + seed4 + seed5 + seed6 + seed7 + seed8
    binaryImg = regionGrow(img_eq,seeds,threshold)
    

    #running on columns
    convert_matrix_copy = binaryImg.copy()
    convert_matrix_copy.shape[1]
    for column_index in range(0,convert_matrix_copy.shape[1]):
        count_one = 0
        index_one = []
        for row_index in range(0,len(convert_matrix_copy)):
            if(convert_matrix_copy[row_index][column_index] == 1):
                index_one.append(row_index)
                count_one += 1
            if(count_one > 1):
                start_index = index_one[0]
                end_index = index_one[len(index_one)-1]
                for i in range(start_index,end_index):
                    convert_matrix_copy[i][column_index] = 1
    
    #running on rows
    convert_matrix_copy1 = convert_matrix_copy.copy()
    for row_index in range(len(convert_matrix_copy1)):
        count_one = 0
        index_one = []
        for element_index in range(0,len(convert_matrix_copy1[row_index])-1):
            if(convert_matrix_copy1[row_index][element_index] == 1):
                index_one.append(element_index)
                count_one += 1
            if(count_one > 1):
                start_index = index_one[0]
                end_index = index_one[len(index_one)-1]
                for i in range(start_index,end_index):
                    convert_matrix_copy1[row_index][i] = 1
    # cv2.imshow('Original image',img)
    # cv2.imshow('Canny image', edges)
    # cv2.imwrite("./postProcessImages/Canny.jpg", edges)
    im2 = np.array(convert_matrix_copy1 * 255, dtype = np.uint8)
    image1 = Image.fromarray(im2)

    image_string = cv2.imencode('.jpg', np.array(image1))[1].tobytes()
    imageName = flask.request.get_json()["filename"] + '_singleregion' + str(num) + '.jpg'

    label_img = skimage.measure.label(convert_matrix_copy1)
    regions = skimage.measure.regionprops(label_img)

    area_2=0
    peri_2=0
    for props in regions:
        area_2=area_2 +props.area
        peri_2=peri_2+ vdk_perimeter(props.convex_image)
        # print (props.area, vdk_perimeter(props.convex_image))
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
        "url": os.environ.get("CLOUDFRONT_URL")+imageName,
        "area": int(area_2),
        "perimeter":int(peri_2)
    }

@app.route('/singleregion1', methods=['GET', 'POST'])
def SingleRegion1():
    return SingleRegion(3, 1)

@app.route('/singleregion2', methods=['GET', 'POST'])
def SingleRegion2():
    return SingleRegion(4, 2)

@app.route('/singleregion3', methods=['GET', 'POST'])
def SingleRegion3():
    return SingleRegion(6, 3)
    