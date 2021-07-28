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

from dotenv import load_dotenv
load_dotenv()

s3 = boto3.client(
   "s3",
   aws_access_key_id=os.environ.get("S3_ACCESS_KEY_ID"),
   aws_secret_access_key=os.environ.get("S3_SECRET_ACCESS_KEY")
)



app = flask.Flask(__name__)


@app.route('/', methods = ['GET', 'POST'])
def handle_request():
    files_ids = list(flask.request.files)
    print("\nNumber of Received Images : ", len(files_ids))
    image_num = 1
    for file_id in files_ids:
        print("\nSaving Image ", str(image_num), "/", len(files_ids))
        imagefile = flask.request.files[file_id]
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        print("Image Filename : " + imagefile.filename)
        timestr  =  time.strftime("%Y%m%d-%H%M%S")
        #Filename1 = timestr+'_'+filename
        #global Filename
       # Filename = "WTF"
        imagefile.save("test")
        image_num = image_num + 1
    print("\n")
    return "Image(s) Uploaded Successfully. Come Back Soon."

# @app.route('/Binary', methods=['GET', 'POST'])
# def Binary():
#     print("Its working canny Binary")
#     image = cv2.imread('/home/jathin/Desktop/Projects/Boils FInal/test')
#     # scale_percent = 20 # percent of original size
#     # width = int(image.shape[1] * scale_percent / 100)
#     # height = int(image.shape[0] * scale_percent / 100)
#     # dim = (width, height)
#     # img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) 
#     ret,thresh1 = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
#     cv2.imshow('Original image',image)
#     cv2.imshow('Gray image', thresh1)
#     # ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
#     # ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
#     # ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
#     # ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
#     # titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
#     # images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
#     # for i in range(6):
#     #     plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
#     #     plt.title(titles[i])
#     #     plt.xticks([]),plt.yticks([])

#     return "Success"

    image_num = 1
    for file_id in files_ids:
        print("\nSaving Image ", str(image_num), "/", len(files_ids))
        imagefile = flask.request.files[file_id]
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        print("Image Filename : " + imagefile.filename)
        timestr  =  time.strftime("%Y%m%d-%H%M%S")
        #Filename1 = timestr+'_'+filename
        #global Filename
       # Filename = "WTF"
        imagefile.save("test")
        image_num = image_num + 1
        image = cv2.imread('/home/jathin/Desktop/Projects/Boils FInal/test')
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('Original image',image)
#     cv2.imshow('Gray image', gray)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     print("Working")
#     return "Success"
@app.route('/process', methods=['GET', 'POST'])
def process():
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
    record = flask.request.get_json()
    # print(record["points"])
    json = record

    x_values = [];
    y_values = [];
    x_y_values = [];
    for value in json['lines']:
        for val in value["points"]:
            x_values.append(round(val['x']))
            y_values.append(round(val['y']))
            x_y_values.append([round(val['x']),round(val['y'])])

    max_x = max(x_values)
    max_y = max(y_values)

    convert_matrix = np.zeros(shape=(max_x+1,max_y+1))
    for value in x_y_values:
        convert_matrix[value[0]][value[1]] = 1 

    convert_matrix_copy = convert_matrix.copy()
    for row_index in range(len(convert_matrix_copy)):
        count_one = 0
        index_one = []
        for element_index in range(0,len(convert_matrix_copy[row_index])-1):
            if(convert_matrix_copy[row_index][element_index] == 1):
                index_one.append(element_index)
                count_one += 1
            if(count_one > 1):
                start_index = index_one[0]
                end_index = index_one[len(index_one)-1]
                for i in range(start_index,end_index):
                    convert_matrix_copy[row_index][i] = 1


    #running on rows
    convert_matrix_copy1 = convert_matrix_copy.copy()
    convert_matrix_copy1.shape[1]
    for column_index in range(0,convert_matrix_copy1.shape[1]):
        count_one = 0
        index_one = []
        for row_index in range(0,len(convert_matrix_copy1)):
            if(convert_matrix_copy1[row_index][column_index] == 1):
                index_one.append(row_index)
                count_one += 1
            if(count_one > 1):
                start_index = index_one[0]
                end_index = index_one[len(index_one)-1]
                for i in range(start_index,end_index):
                    convert_matrix_copy1[i][column_index] = 1

    label_img = skimage.measure.label(convert_matrix_copy1)
    regions = skimage.measure.regionprops(label_img)

    peri_1=0
    area_1=0
    for props in regions:
        area_1+=props.area
        peri_1+=vdk_perimeter(props.convex_image)

    return {"perimeter": int(peri_1), "area":int(area_1)}

@app.route('/sobelx', methods=['GET', 'POST'])
def SingleWatershed():
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
    # print(flask.request.get_json()["url"])
    req = urllib.request.urlopen(flask.request.get_json()["url"])
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(arr, 0)
    # image = cv2.imread('./preProcessImages/test.png')
     
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_equalized = cv2.equalizeHist(image)
    markers = np.zeros_like(image_equalized)
    markers[image_equalized < 150] = 1
    markers[image_equalized > 230] = 2
    elevation_map = sobel(image_equalized)

    markers = np.zeros_like(image_equalized)
    markers[image_equalized < 150] = 1
    markers[image_equalized > 230] = 2
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
    imageName = flask.request.get_json()["filename"] + '_sobelX.jpg'
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

# @app.route('/sobelx', methods=['GET', 'POST'])
# def SobelX():
#     # print(flask.request.get_json()["url"])
#     req = urllib.request.urlopen(flask.request.get_json()["url"])
#     arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
#     image = cv2.imdecode(arr, -1)
#     # image = cv2.imread('./preProcessImages/test.png')
#     scale_percent = 100 # percent of original size
#     width = int(image.shape[1] * scale_percent / 100)
#     height = int(image.shape[0] * scale_percent / 100)
#     dim = (width, height)
#     img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) 
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     image_equalized = cv2.equalizeHist(gray)
#     blur = cv2.GaussianBlur(image_equalized,(3,3),0)
#     edges = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
#     peri_2 = np.count_nonzero(edges == 0)
#     print(peri_2)
#     area_2 = np.count_nonzero(edges == 255)
#     print(area_2)
   

#     # cv2.imshow('Original image',img)
#     # cv2.imshow('Canny image', edges)
#     # cv2.imwrite("./postProcessImages/Canny.jpg", edges)
#     image_string = cv2.imencode('.jpg', edges)[1].tobytes()
#     imageName = flask.request.get_json()["filename"] + '_sobelX.jpg'
#     # cv2.imshow('Gray image', gray)
#     try:

#         s3.upload_fileobj(
#             io.BytesIO(image_string),
#             os.environ.get("S3_BUCKET_NAME"),
#             imageName,
#             ExtraArgs={
#                 "ACL": "public-read",
#                 "ContentType": 'image/jpeg'
#             }
#         )

#     except Exception as e:
#         # This is a catch all exception, edit this part to fit your needs.
#         print("Something Happened: ", e)
#         return e
#     print(os.environ.get("CLOUDFRONT_URL")+imageName)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     print("Working")
#     return {
#         "url": os.environ.get("CLOUDFRONT_URL") + imageName,
#         "perimeter": area_2,
#         "area":peri_2
#     }

@app.route('/sobely', methods=['GET', 'POST'])
def SingleWatershedRegion():

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
    # print(flask.request.get_json()["url"])
    req = urllib.request.urlopen(flask.request.get_json()["url"])
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(arr, 0)
    # image = cv2.imread('./preProcessImages/test.png')

    img_eq = cv2.equalizeHist(image)

    height1, weight1 = img_eq.shape
    h1 = int(height1/2)
    w1 = int(weight1/2)
    #seeds = (h1, w1)

    #seed1 = [Point(h1, w1), Point(h1+2, w1+2), Point(h1+3, w1+3), Point(h1+4, w1+4), Point(h1+5, w1+5), Point(h1+6, w1+6), Point(h1+7, w1+7), Point(h1+8, w1+8)]
    seed2 = [Point(h1-24, w1-24), Point(h1-20, w1-20), Point(h1-36, w1-36), Point(h1-14, w1-14), Point(h1-10, w1-10), Point(h1-6, w1-6), Point(h1-7, w1-7), Point(h1-8, w1-8), Point(h1-5, w1-5)]
    #seed3 = [Point(h1, -w1), Point(h1+5, w1-5), Point(h1+5, w1-8), Point(h1+2, w1-8), Point(h1+2, w1+8)]
    seed4 = [Point(h1-5, w1+5), Point(h1-5, w1+8), Point(h1+5, w1-5), Point(h1+5, w1-8),Point(h1-5, w1-2), Point(h1-5, w1-5), Point(h1-5, w1-8) ]
    seed5 = [Point(h1-14, w1+5), Point(h1-14, w1+2), Point(h1-14, w1+1), Point(h1-14, w1+3), Point(h1-14, w1+4), Point(h1-10, w1+5), Point(h1-10, w1+2), Point(h1-10, w1+1), Point(h1-10, w1+3), Point(h1-10, w1+4)]
    #seed6 = [Point(h1+24, w1+24), Point(h1+20, w1+20), Point(h1+36, w1+36)]
    seed7 = [Point(h1-15, w1+15), Point(h1-15, w1+18), Point(h1+15, w1-15), Point(h1+15, w1-18),Point(h1-15, w1-12), Point(h1-15, w1-15), Point(h1-15, w1-18) ]
    seed8 = [Point(h1-24, w1+15), Point(h1-24, w1+12), Point(h1-24, w1+10), Point(h1-24, w1+13), Point(h1-24, w1+14), Point(h1-20, w1+15), Point(h1-20, w1+12), Point(h1-20, w1+10), Point(h1-20, w1+13), Point(h1-20, w1+14)]

    seeds = seed2 # + seed4 + seed5 + seed7 + seed8
    binaryImg = regionGrow(img_eq,seeds,4)


    markers = np.zeros_like(img_eq)
    markers[img_eq < 20] = 1
    markers[img_eq > 250] = 2
    elevation_map = sobel(img_eq)

    markers = np.zeros_like(img_eq)
    markers[img_eq < 20] = 1
    markers[img_eq > 250] = 2
    segmentation = watershed(elevation_map, markers)
    segmentation = ndi.binary_fill_holes(segmentation - 1)

    comb = np.logical_or(segmentation, binaryImg)

    #running on columns
    convert_matrix_copy = comb.copy()
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

    im1 = np.array(convert_matrix_copy1 * 255, dtype = np.uint8)
    image = Image.fromarray(im1)

    label_img = skimage.measure.label(convert_matrix_copy1)
    regions = skimage.measure.regionprops(label_img)

    area_2=0
    peri_2=0
    for props in regions:
        area_2+=props.area
        peri_2+=vdk_perimeter(props.convex_image)
        # print (props.area, vdk_perimeter(props.convex_image))
    
    image_string = cv2.imencode('.jpg', np.array(image))[1].tobytes()
    imageName = flask.request.get_json()["filename"] + '_sobelY.jpg'
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
        "perimeter": int(peri_2),
        "area":int(area_2)
    }

# @app.route('/sobely', methods=['GET', 'POST'])
# def SobelY():
#     # print(flask.request.get_json()["url"])
#     req = urllib.request.urlopen(flask.request.get_json()["url"])
#     arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
#     image = cv2.imdecode(arr, -1)
#     # image = cv2.imread('./preProcessImages/test.png')
#     scale_percent = 100 # percent of original size
#     width = int(image.shape[1] * scale_percent / 100)
#     height = int(image.shape[0] * scale_percent / 100)
#     dim = (width, height)
#     img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) 
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     image_equalized = cv2.equalizeHist(gray)
#     blur = cv2.GaussianBlur(image_equalized,(3,3),0)
#     edges = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
#     peri_2 = np.count_nonzero(edges == 0)
#     print(peri_2)
#     area_2 = np.count_nonzero(edges == 255)
#     print(area_2)
#     # cv2.imshow('Original image',img)
#     # cv2.imshow('Canny image', edges)
#     # cv2.imwrite("./postProcessImages/Canny.jpg", edges)
#     image_string = cv2.imencode('.jpg', edges)[1].tobytes()
#     imageName = flask.request.get_json()["filename"] + '_sobelY.jpg'
#     # cv2.imshow('Gray image', gray)
#     try:

#         s3.upload_fileobj(
#             io.BytesIO(image_string),
#             os.environ.get("S3_BUCKET_NAME"),
#             imageName,
#             ExtraArgs={
#                 "ACL": "public-read",
#                 "ContentType": 'image/jpeg'
#             }
#         )

#     except Exception as e:
#         # This is a catch all exception, edit this part to fit your needs.
#         print("Something Happened: ", e)
#         return e
#     print(os.environ.get("CLOUDFRONT_URL")+imageName)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     print("Working")
#     return {
#         "url": os.environ.get("CLOUDFRONT_URL")+imageName,
#         "perimeter": area_2,
#         "area":peri_2
#     }

@app.route('/canny1', methods=['GET', 'POST'])
def SingleRegion1():
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
    binaryImg = regionGrow(img_eq,seeds,3)
    

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
    imageName = flask.request.get_json()["filename"] + '_canny1.jpg'

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

@app.route('/canny2', methods=['GET', 'POST'])
def SingleRegion2():
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
    binaryImg = regionGrow(img_eq,seeds,4)
    

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
    imageName = flask.request.get_json()["filename"] + '_canny2.jpg'

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

@app.route('/canny3', methods=['GET', 'POST'])
def SingleRegion3():
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
    binaryImg = regionGrow(img_eq,seeds,6)
    

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
    imageName = flask.request.get_json()["filename"] + '_canny3.jpg'

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

@app.route('/otsu', methods=['GET', 'POST'])
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
        # print (props.area, vdk_perimeter(props.convex_image))

    # dim = (width, height)
    # img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) 
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # image_equalized = cv2.equalizeHist(gray)
    # blur = cv2.GaussianBlur(image_equalized,(5,5),0)
    # ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # peri_2 = np.count_nonzero(th3 == 0)
    # print(peri_2)
    # area_2 = np.count_nonzero(th3 == 255)
    # print(area_2)
    # # cv2.imshow('Original image',img)
    # # cv2.imshow('Canny image', edges)
    # # cv2.imwrite("./postProcessImages/Canny.jpg", edges)
    # image_string = cv2.imencode('.jpg', th3)[1].tobytes()
    # imageName = flask.request.get_json()["filename"] + '_otsu.jpg'
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
        "perimeter": int(perimeter),
        "area":int(area)
    }

@app.route('/laplacian', methods=['GET', 'POST'])
def Laplacian():
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
    blur = cv2.GaussianBlur(image_equalized,(3,3),0)
    edges = cv2.Laplacian(blur, cv2.CV_64F)
    peri_2 = np.count_nonzero(edges == 0)
    print(peri_2)
    area_2 = np.count_nonzero(edges == 255)
    print(area_2)
    # cv2.imshow('Original image',img)
    # cv2.imshow('Canny image', edges)
    # cv2.imwrite("./postProcessImages/Canny.jpg", edges)
    image_string = cv2.imencode('.jpg', edges)[1].tobytes()
    imageName = flask.request.get_json()["filename"] + '_laplacian.jpg'
    # cv2.imshow('Gray image', gray)
    print(type(image_string))
    print(type(io.BytesIO(image_string)))
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
        "perimeter": area_2,
        "area":peri_2
    }


app.run(port=4000, debug=True)