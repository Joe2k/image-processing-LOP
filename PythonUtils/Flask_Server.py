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

import numpy as np
from matplotlib import pyplot as plt

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
    record = flask.request.get_json()
    # print(record["points"])
    json = record

    x_values = []
    y_values = []
    x_y_values = []
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

    #convert matrix = 1's at (x,y) cordinates
    print(convert_matrix)

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

    #convert matrix copy = 1's at (x,y) cordinates + 1's inside the region selected
    print(convert_matrix_copy)

    peri_1 = np.count_nonzero(convert_matrix == 1.0)
    print(peri_1)
    area_1 = np.count_nonzero(convert_matrix_copy == 1.0)
    print(area_1)
    return {"perimeter": peri_1, "area":area_1}

@app.route('/sobelx', methods=['GET', 'POST'])
def SobelX():
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
    edges = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    peri_2 = np.count_nonzero(edges == 0)
    print(peri_2)
    area_2 = np.count_nonzero(edges == 255)
    print(area_2)
   

    # cv2.imshow('Original image',img)
    # cv2.imshow('Canny image', edges)
    # cv2.imwrite("./postProcessImages/Canny.jpg", edges)
    image_string = cv2.imencode('.jpg', edges)[1].tobytes()
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
        "perimeter": area_2,
        "area":peri_2
    }

@app.route('/sobely', methods=['GET', 'POST'])
def SobelY():
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
    edges = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    peri_2 = np.count_nonzero(edges == 0)
    print(peri_2)
    area_2 = np.count_nonzero(edges == 255)
    print(area_2)
    # cv2.imshow('Original image',img)
    # cv2.imshow('Canny image', edges)
    # cv2.imwrite("./postProcessImages/Canny.jpg", edges)
    image_string = cv2.imencode('.jpg', edges)[1].tobytes()
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
        "perimeter": area_2,
        "area":peri_2
    }



@app.route('/canny', methods=['GET', 'POST'])
def Canny():
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
    edges = cv2.Canny(image_equalized, 100, 200)
    peri_2 = np.count_nonzero(edges == 0)
    print(peri_2)
    area_2 = np.count_nonzero(edges == 255)
    print(area_2)
    # cv2.imshow('Original image',img)
    # cv2.imshow('Canny image', edges)
    # cv2.imwrite("./postProcessImages/Canny.jpg", edges)
    image_string = cv2.imencode('.jpg', edges)[1].tobytes()
    imageName = flask.request.get_json()["filename"] + '_canny.jpg'
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
        "perimeter": area_2,
        "area":peri_2
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
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    peri_2 = np.count_nonzero(th3 == 0)
    print(peri_2)
    area_2 = np.count_nonzero(th3 == 255)
    print(area_2)
    # cv2.imshow('Original image',img)
    # cv2.imshow('Canny image', edges)
    # cv2.imwrite("./postProcessImages/Canny.jpg", edges)
    image_string = cv2.imencode('.jpg', th3)[1].tobytes()
    imageName = flask.request.get_json()["filename"] + '_otsu.jpg'
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
        "perimeter": peri_2,
        "area":area_2
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