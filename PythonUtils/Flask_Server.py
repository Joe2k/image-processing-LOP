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
import single_watershed
import single_watershed_region
import otsu
import single_region
# Process API for calculating original Area and Perimeter
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


app.run(port=4000, debug=True)