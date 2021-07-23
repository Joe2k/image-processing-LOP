# -*- coding: utf-8 -*-
"""single_manual.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10ltnKrjKK75v9rkrFyomQgkj10Ys2CgV
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from numpy import asarray
import skimage.measure
import skimage.morphology

#calculating peri from the contour code used in image processing on area image
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

json = {"lines":[{"points":[{"x":147,"y":118},{"x":147,"y":118},{"x":147,"y":119},{"x":147,"y":121},{"x":146,"y":122},{"x":145,"y":122},{"x":144,"y":124},{"x":143,"y":125},{"x":141,"y":126},{"x":140,"y":127},{"x":139,"y":129},{"x":137,"y":130},{"x":135,"y":131},{"x":130,"y":132},{"x":129,"y":133},{"x":125,"y":135},{"x":122,"y":136},{"x":119,"y":137},{"x":116,"y":137},{"x":114,"y":139},{"x":113,"y":139},{"x":111,"y":139},{"x":110,"y":139},{"x":107,"y":139},{"x":105,"y":140},{"x":104,"y":140},{"x":100,"y":140},{"x":97,"y":141},{"x":96,"y":141},{"x":95,"y":141},{"x":94,"y":141},{"x":93,"y":141},{"x":92,"y":141},{"x":90,"y":141},{"x":89,"y":141},{"x":85,"y":141},{"x":84,"y":141},{"x":82,"y":141},{"x":81,"y":141},{"x":80,"y":141},{"x":77,"y":141},{"x":76,"y":141},{"x":74,"y":141},{"x":72,"y":140},{"x":71,"y":140},{"x":70,"y":139},{"x":68,"y":138},{"x":66,"y":137},{"x":65,"y":136},{"x":64,"y":136},{"x":62,"y":136},{"x":61,"y":135},{"x":60,"y":134},{"x":58,"y":133},{"x":55,"y":131},{"x":53,"y":130},{"x":52,"y":129},{"x":51,"y":128},{"x":51,"y":127},{"x":50,"y":126},{"x":50,"y":124},{"x":50,"y":123},{"x":50,"y":122},{"x":50,"y":121},{"x":50,"y":119},{"x":50,"y":118},{"x":50,"y":116},{"x":50,"y":115},{"x":50,"y":113},{"x":51,"y":111},{"x":52,"y":109},{"x":53,"y":108},{"x":54,"y":106},{"x":54,"y":105},{"x":56,"y":103},{"x":56,"y":101},{"x":56,"y":99},{"x":57,"y":98},{"x":59,"y":96},{"x":60,"y":93},{"x":61,"y":91},{"x":63,"y":89},{"x":63,"y":88},{"x":64,"y":87},{"x":65,"y":86},{"x":66,"y":84},{"x":67,"y":83},{"x":69,"y":81},{"x":70,"y":80},{"x":72,"y":80},{"x":73,"y":78},{"x":75,"y":77},{"x":77,"y":76},{"x":79,"y":75},{"x":79,"y":74},{"x":81,"y":74},{"x":82,"y":73},{"x":83,"y":73},{"x":85,"y":72},{"x":85,"y":71},{"x":87,"y":71},{"x":88,"y":70},{"x":89,"y":69},{"x":90,"y":69},{"x":90,"y":68},{"x":92,"y":68},{"x":93,"y":67},{"x":97,"y":66},{"x":98,"y":65},{"x":99,"y":64},{"x":101,"y":63},{"x":103,"y":62},{"x":105,"y":61},{"x":106,"y":61},{"x":109,"y":61},{"x":110,"y":61},{"x":112,"y":59},{"x":113,"y":59},{"x":114,"y":59},{"x":115,"y":59},{"x":116,"y":59},{"x":118,"y":59},{"x":119,"y":59},{"x":120,"y":59},{"x":122,"y":59},{"x":124,"y":59},{"x":126,"y":59},{"x":128,"y":59},{"x":129,"y":59},{"x":131,"y":59},{"x":133,"y":59},{"x":134,"y":59},{"x":137,"y":59},{"x":140,"y":60},{"x":141,"y":60},{"x":143,"y":61},{"x":145,"y":61},{"x":146,"y":62},{"x":147,"y":63},{"x":148,"y":63},{"x":149,"y":65},{"x":150,"y":66},{"x":152,"y":67},{"x":152,"y":68},{"x":153,"y":69},{"x":154,"y":71},{"x":155,"y":73},{"x":155,"y":74},{"x":156,"y":75},{"x":156,"y":76},{"x":156,"y":77},{"x":157,"y":78},{"x":157,"y":79},{"x":157,"y":81},{"x":157,"y":82},{"x":157,"y":84},{"x":157,"y":85},{"x":157,"y":86},{"x":157,"y":88},{"x":157,"y":89},{"x":157,"y":91},{"x":157,"y":92},{"x":157,"y":94},{"x":157,"y":96},{"x":156,"y":98},{"x":156,"y":99},{"x":156,"y":100},{"x":155,"y":101},{"x":154,"y":102},{"x":154,"y":103},{"x":153,"y":104},{"x":152,"y":105},{"x":151,"y":107},{"x":151,"y":108},{"x":150,"y":109},{"x":149,"y":110},{"x":148,"y":111},{"x":147,"y":112},{"x":147,"y":113},{"x":146,"y":114},{"x":145,"y":114},{"x":145,"y":115},{"x":144,"y":115},{"x":144,"y":117},{"x":143,"y":118},{"x":142,"y":118},{"x":141,"y":120},{"x":140,"y":121},{"x":139,"y":122},{"x":139,"y":124},{"x":139,"y":125},{"x":138,"y":126},{"x":138,"y":127},{"x":138,"y":129},{"x":138,"y":129}],"brushColor":"#fff","brushRadius":2}],"width":200,"height":200}

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

  for props in regions:
      print (props.area, vdk_perimeter(props.convex_image))